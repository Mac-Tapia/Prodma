"""Add the source-plan monthly forecasters to the local notebook once."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebook" / "proyecciondemanda.ipynb"
TAG = "prodma_extended_forecasters_v1"


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

# Perfil GPU local: se mantiene idempotente y se prueba antes de entrenar datos reales.
config_source = "".join(nb["cells"][5]["source"])
config_source = config_source.replace(
    "CFG = dict(modo='rapido', seed=42,",
    "CFG = dict(modo='rapido', backend='auto', seed=42,",
)
nb["cells"][5]["source"] = config_source.splitlines(keepends=True)

deps_source = "".join(nb["cells"][3]["source"])
deps_source = deps_source.replace(
    "'joblib':'joblib'",
    "'joblib':'joblib', 'prophet':'prophet', 'torch':'torch'",
)
nb["cells"][3]["source"] = deps_source.splitlines(keepends=True)

models_source = "".join(nb["cells"][27]["source"])
if "def probar_backends_gpu():" not in models_source:
    marker = "PACIENCIA_EARLY_STOPPING=20\n"
    gpu_probe = """PACIENCIA_EARLY_STOPPING=20

def probar_backends_gpu():
    solicitado=CFG.get('backend','auto')
    if solicitado not in ('auto','cpu','gpu'):
        raise ValueError("CFG['backend'] debe ser auto, cpu o gpu")
    estado={'XGBoost':False,'CatBoost':False,'LightGBM':False};detalle={}
    if solicitado=='cpu':
        return estado,{k:'CPU solicitado' for k in estado}
    Xp=np.arange(128,dtype=float).reshape(64,2);yp=np.sin(np.arange(64)/10)
    pruebas={
      'XGBoost':XGBRegressor(n_estimators=4,max_depth=2,tree_method='hist',device='cuda:0',verbosity=0),
      'CatBoost':CatBoostRegressor(iterations=4,depth=2,task_type='GPU',devices='0',verbose=False),
      'LightGBM':LGBMRegressor(n_estimators=4,max_depth=2,device_type='gpu',gpu_device_id=0,verbosity=-1)}
    for nombre,modelo in pruebas.items():
        try:
            modelo.fit(Xp,yp);pred=modelo.predict(Xp[:2])
            if not np.isfinite(pred).all():raise ValueError('predicción no finita')
            estado[nombre]=True;detalle[nombre]='GPU verificada con fit/predict real'
        except Exception as exc:
            detalle[nombre]=f'{type(exc).__name__}: {exc}'
    if solicitado=='gpu' and not (estado['XGBoost'] and estado['CatBoost']):
        raise RuntimeError('GPU obligatoria no disponible para XGBoost/CatBoost: '+str(detalle))
    return estado,detalle

GPU_OK,GPU_DETALLE=probar_backends_gpu()
pd.DataFrame([{'modelo':k,'gpu_disponible':GPU_OK[k],'detalle':GPU_DETALLE[k]} for k in GPU_OK]).to_csv(
    RUN/'logs/backends_gpu.csv',index=False)
print('Backends GPU efectivos:',GPU_OK)
"""
    models_source = models_source.replace(marker, gpu_probe)
    models_source = models_source.replace(
        "random_state=seed,tree_method='hist',\n                    early_stopping_rounds=",
        "random_state=seed,tree_method='hist',device=('cuda:0' if GPU_OK['XGBoost'] else 'cpu'),\n                    early_stopping_rounds=",
    )
    models_source = models_source.replace(
        "LGBMRegressor(n_estimators=400,n_jobs=2,random_state=seed,verbosity=-1)",
        "LGBMRegressor(n_estimators=400,n_jobs=2,random_state=seed,verbosity=-1,device_type=('gpu' if GPU_OK['LightGBM'] else 'cpu'))",
    )
    models_source = models_source.replace(
        "CatBoostRegressor(iterations=400,thread_count=2,random_seed=seed,verbose=False)",
        "CatBoostRegressor(iterations=400,thread_count=2,random_seed=seed,verbose=False,task_type=('GPU' if GPU_OK['CatBoost'] else 'CPU'),devices=('0' if GPU_OK['CatBoost'] else None))",
    )
nb["cells"][27]["source"] = models_source.splitlines(keepends=True)

if not any(cell.get("metadata", {}).get("prodma_tag") == TAG for cell in nb["cells"]):
    nb["cells"].extend(
        [
            {
                "cell_type": "markdown",
                "metadata": {"prodma_tag": TAG},
                "source": [
                    "## 26 · Modelos adicionales exigidos por el plan\n",
                    "\n",
                    "Compara en resolución mensual el ingenuo estacional, ARIMA, SARIMA, Holt‑Winters, Prophet y LSTM. SARIMAX se habilita únicamente cuando se entregan regresores históricos y futuros alineados. Los modelos mensuales no se mezclan con el ranking operacional de 15 minutos.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"prodma_tag": TAG},
                "outputs": [],
                "source": """import sys
SRC_LOCAL=RAIZ_PROYECTO/'src'
if str(SRC_LOCAL) not in sys.path: sys.path.insert(0,str(SRC_LOCAL))
from prodma.forecast_models import create_forecaster

def comparar_modelos_mensuales(serie,horizonte=12,exog=None,device='auto'):
    serie=serie.replace([np.inf,-np.inf],np.nan).dropna().astype(float)
    if len(serie)<max(36,horizonte+24):
        raise ValueError('Se requieren al menos 36 meses y dos ciclos antes del test')
    train,test=serie.iloc[:-horizonte],serie.iloc[-horizonte:]
    exog_train=exog.iloc[:-horizonte] if exog is not None else None
    exog_test=exog.iloc[-horizonte:] if exog is not None else None
    nombres=['SeasonalNaive','ARIMA','SARIMA','HoltWinters','Prophet','LSTM']
    if exog is not None:nombres.insert(4,'SARIMAX')
    filas=[];predicciones={}
    for nombre in nombres:
        started=time.time()
        try:
            kwargs={'seasonal_periods':12}
            if nombre=='LSTM':kwargs.update(device=device,epochs=80,hidden_size=32,seed=CFG['seed'])
            model=create_forecaster(nombre,**kwargs)
            model.fit(train,exog=exog_train if nombre=='SARIMAX' else None)
            pred=model.predict(horizonte,exog=exog_test if nombre=='SARIMAX' else None)
            predicciones[nombre]=pd.Series(pred,index=test.index)
            filas.append(dict(modelo=nombre,estado='OK',backend=getattr(model,'device_','CPU'),
                segundos=time.time()-started,**metrics(test,pred)))
        except Exception as exc:
            filas.append(dict(modelo=nombre,estado='ERROR',backend=None,
                segundos=time.time()-started,error=f'{type(exc).__name__}: {exc}'))
    return pd.DataFrame(filas),pd.DataFrame(predicciones,index=test.index)

MODELOS_MENSUALES=[]
if TARGETS:
    for medidor in TARGETS:
        mensual=load_meter(medidor)['P'].resample('MS').mean().dropna()
        try:
            tabla,pred=comparar_modelos_mensuales(mensual,horizonte=12,device='auto')
            tabla.insert(0,'medidor',medidor);MODELOS_MENSUALES.append(tabla)
            pred.to_csv(RUN/'pronosticos'/f'mensual_modelos_{TOKENS[medidor]}.csv')
        except Exception as exc:
            log('modelos_mensuales',medidor,'ERROR',exc)
if MODELOS_MENSUALES:
    metricas_mensuales=pd.concat(MODELOS_MENSUALES,ignore_index=True)
    metricas_mensuales.to_csv(RUN/'tablas/metricas_modelos_mensuales.csv',index=False)
    display(metricas_mensuales)
""".splitlines(keepends=True),
            },
        ]
    )

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    source = source.replace("demanda2023.xlsxxx", "demanda2023.xlsx")
    cell["source"] = source.splitlines(keepends=True)

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(NOTEBOOK)
