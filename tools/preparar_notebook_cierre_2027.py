"""Aplica al notebook los cambios del cierre 2027 (reanudación sin reentrenar, horizonte a
fin de 2027, pronóstico recursivo rápido/paralelo, guardas de sanidad, figuras reanudables).

Uso (desde la raíz del proyecto):
    .\\.venv\\Scripts\\python.exe tools\\preparar_notebook_cierre_2027.py

Cada edición busca un ANCLA exacta y falla con un mensaje claro si no la encuentra; así el
script no puede dejar el notebook a medias sin avisar. Es idempotente: si el notebook ya
tiene los cambios (marca ``prodma.reanudar``), no hace nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebook" / "proyecciondemanda.ipynb"
OMITIR = {"prodma": {"omitir_al_reanudar": True}}
SOLO_REANUDAR = {"prodma": {"solo_al_reanudar": True}}


def celda_con(nb, ancla: str, tipo: str = "code") -> int:
    hallados = [i for i, c in enumerate(nb.cells) if c.cell_type == tipo and ancla in c.source]
    if len(hallados) != 1:
        raise SystemExit(f"Ancla {ancla!r}: se esperaba 1 celda y hay {len(hallados)} ({hallados})")
    return hallados[0]


def reemplazar(nb, i: int, viejo: str, nuevo: str, veces: int = 1) -> None:
    fuente = nb.cells[i].source
    if fuente.count(viejo) != veces:
        raise SystemExit(f"Celda {i}: {viejo[:70]!r} aparece {fuente.count(viejo)} veces (se esperaban {veces})")
    nb.cells[i].source = fuente.replace(viejo, nuevo)


def nueva_codigo(fuente: str, metadata: dict | None = None):
    c = nbformat.v4.new_code_cell(fuente)
    c.metadata.update(metadata or {})
    return c


# ---------------------------------------------------------------------------
# Texto de las celdas nuevas / reescritas
# ---------------------------------------------------------------------------
CELDA_RUN = '''RAIZ_PROYECTO=encontrar_raiz_proyecto()
DIRECTORIO_HISTORICO=RAIZ_PROYECTO/'datahistorica'
DIRECTORIO_CLIENTES=RAIZ_PROYECTO/'nuevosclientes'
ROOT=RAIZ_PROYECTO/'resultados'; ROOT.mkdir(parents=True,exist_ok=True)
CACHE=ROOT/'cache_v1'; CACHE.mkdir(parents=True,exist_ok=True)
# Nombre descriptivo: <modo>-<backend>_YYYYMMDD_HHMMSS (ej: R-AUT_20260917_114523)
modo_tag = CFG.get('modo', 'rapido')[0].upper()  # R=Rapido, C=Completo
backend_tag = CFG.get('backend', 'auto')[:3].upper()  # AUT/CPU/GPU
# Reanudación: con CFG['reanudar_run'] se CONTINÚA en la carpeta de una corrida previa (sus
# modelos/tablas/figuras se reutilizan, nada se reentrena) en vez de crear una nueva.
REANUDANDO=bool(CFG.get('reanudar_run'))
if REANUDANDO:
    RUN=Path(CFG['reanudar_run'])
    if not RUN.is_absolute(): RUN=RAIZ_PROYECTO/RUN
    assert (RUN/'modelos').is_dir() and (RUN/'tablas/metricas.csv').exists(), f'{RUN} no parece una corrida previa'
else:
    RUN=ROOT/(f'{modo_tag}-{backend_tag}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}')
for p in ('tablas','figuras','modelos','pronosticos','logs'):
    (RUN/p).mkdir(parents=True,exist_ok=True)
if REANUDANDO:
    (RUN/f'config_reanudacion_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json').write_text(
        json.dumps(CFG,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
else:
    (RUN/'config.json').write_text(json.dumps(CFG,indent=2,ensure_ascii=False),encoding='utf-8')
    (RUN/'entorno.txt').write_text(
        subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
LOG=[]
if REANUDANDO and (RUN/'logs/eventos.csv').exists():
    LOG=pd.read_csv(RUN/'logs/eventos.csv').fillna('').to_dict('records')  # conserva los eventos previos
def log(stage,item,status,detail=''):
    LOG.append(dict(etapa=stage,item=str(item),estado=status,detalle=str(detail)))
    pd.DataFrame(LOG).to_csv(RUN/'logs/eventos.csv',index=False)
print('Proyecto:',RAIZ_PROYECTO)
print('Resultados:',RUN)
if REANUDANDO: print('REANUDACIÓN sobre una corrida previa: no se reentrena ningún modelo ML.')
'''

CELDA_RESTAURAR = '''# --- 14b · Restauración del estado desde una corrida previa (solo con CFG['reanudar_run']) ---
# Reemplaza a las secciones 14-16 al reanudar: sus resultados ya están en disco
# (modelos/*.joblib, tablas/metricas.csv, comparacion_completa.csv, OLS_referencia.csv).
# NO se entrena nada: se reconstruyen PRED/WINNERS y se VALIDA que cada modelo guardado
# reproduzca su RMSE guardado (si no, se aborta: la restauración no sería fiel).
from prodma.reanudar import (restaurar_estado,construir_ganadores,guardar_estado,cargar_estado,
    ESTADO_OK,ESTADO_CONSTANTE,ESTADO_SIN_DATOS)
if not CFG.get('reanudar_run'):
    print('Sin reanudación: el estado (PRED/WINNERS) proviene del entrenamiento de la sección 14.')
else:
    _previo=cargar_estado(RUN)
    if _previo is None:
        _estado=restaurar_estado(RUN,cfg=CFG,tokens=TOKENS,load_meter=load_meter,features=features,metrics=metrics)
        _estado['validacion'].to_csv(RUN/'tablas/validacion_restauracion.csv',index=False)
        WINNERS=construir_ganadores(pd.DataFrame(_estado['results']),_estado,RUN,TOKENS,
                                    max_dias_sin_datos=CFG['max_dias_sin_datos'])
        _ruta=guardar_estado(RUN,_estado,WINNERS)   # checkpoint: las próximas reanudaciones tardan segundos
        print('Estado restaurado y guardado en',_ruta)
    else:
        _estado,WINNERS=_previo
        print('Estado restaurado desde el checkpoint',RUN/'tablas'/'estado_restaurado.joblib')
    RESULTS=_estado['results'];PRED=_estado['pred']
    _v=_estado['validacion']
    print(f"Validación: {len(_v)} modelos/benchmarks reproducidos; dif. relativa máx. de RMSE = {_v.dif_rel.max():.2e}")
    ESTADO_SERIES=pd.DataFrame([dict(medidor=m,variable=v,modelo_operativo=w['name'],cv_rmse=w['cv_rmse'],
        ultimo_dato=w['last'],estado=w['estado'],motivo=w['motivo']) for (m,v),w in WINNERS.items()])
    ESTADO_SERIES.to_csv(RUN/'tablas/estado_series.csv',index=False)
    print(ESTADO_SERIES.estado.value_counts().to_string())
    for (m,v),w in WINNERS.items():
        if w['estado']!=ESTADO_OK: log('clasificacion_series',f'{m}/{v}',w['estado'],w['motivo'])
'''

CELDA_FIGURAS = '''import gc,psutil,matplotlib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot
# Backend sin GUI durante las ~1300 figuras (con 'inline' cada figura se registra además en
# el kernel) y liberación explícita de memoria: la versión anterior terminó con MemoryError
# tras ~890 figuras. Las figuras que ya existen en disco (p. ej. de una corrida interrumpida)
# se omiten, así que esta celda se puede volver a ejecutar hasta completar.
_backend_previo=matplotlib.get_backend();plt.switch_backend('Agg')
_proceso=psutil.Process();_nuevas=_omitidas=0
try:
    for (m,var,name),d in PRED.items():
        destino=RUN/'figuras'/f'{TOKENS[m]}_{var}_{name}.png'
        if destino.exists(): _omitidas+=1;continue
        e=d.real-d.pred;fig,ax=plt.subplots(4,2,figsize=(15,14))
        d.tail(672).plot(ax=ax[0,0]);ax[0,1].plot(e.index,e)
        ax[1,0].scatter(d.pred,e,s=2);ax[1,1].hist(e,bins=40)
        probplot(e,plot=ax[2,0])
        if e.std()>1e-10 and len(e)>100:
            plot_acf(e,lags=min(96,len(e)//5),ax=ax[2,1])
            plot_pacf(e,lags=min(48,len(e)//5),ax=ax[3,0],method='ywm')
        else:
            ax[2,1].text(.1,.5,'ACF N/A: residuos constantes o insuficientes')
            ax[3,0].text(.1,.5,'PACF N/A')
        e.abs().groupby(e.index.hour).mean().plot(ax=ax[3,1])
        fig.suptitle(f'{m} — {var} ({"kW" if var=="P" else "kvar"}) — {name}; test observado')
        fig.tight_layout();fig.savefig(destino,dpi=120);plt.close(fig);_nuevas+=1
        if _nuevas%50==0:
            gc.collect()
            print(f'  {_nuevas} figuras nuevas ({_omitidas} ya existían); RSS del kernel {_proceso.memory_info().rss/2**20:.0f} MB',flush=True)
finally:
    plt.close('all');plt.switch_backend(_backend_previo)
print(f'Figuras de diagnóstico: {_nuevas} nuevas, {_omitidas} ya existentes; RSS final {_proceso.memory_info().rss/2**20:.0f} MB')
'''

CELDA_PRONOSTICO = r'''import matplotlib
from prodma.recursivo import (features_calendario,proyectar_series,proyectar_recursivo,
    preparar_para_inferencia,cargar_modelo,es_columna_dinamica,BUFFER_MINIMO)
from prodma.sanidad import evaluar_sanidad,calendar_profile
from prodma.reanudar import clasificar_serie,ESTADO_OK,ESTADO_CONSTANTE,ESTADO_SIN_DATOS
BUFFER_HISTORIA=BUFFER_MINIMO+96   # 4 semanas + 1 día: alcance máximo de las features dinámicas
FIN_PLAN=pd.Timestamp(CFG['fin_plan'])
FORECASTS={};MENSUAL={};SANIDAD=[]

# --- Estado de cada serie: solo las ACTIVAS se proyectan con su modelo ML. Las constantes
# se proyectan como constante declarada y las que dejaron de reportar NO se proyectan
# (proyectar años sobre un medidor apagado sería inventar demanda). ---
_ref_ultimo=max(w['last'] for w in WINNERS.values())
for (m,var),w in WINNERS.items():
    if 'estado' not in w: w['estado'],w['motivo']=clasificar_serie(w['cv_rmse'],w['last'],_ref_ultimo,CFG['max_dias_sin_datos'])
    if not w.get('ruta_modelo'): w['ruta_modelo']=str(RUN/'modelos'/f"{TOKENS[m]}_{var}_{w['name']}.joblib")
ACTIVAS=[k for k,w in WINNERS.items() if w['estado']==ESTADO_OK]
CONSTANTES=[k for k,w in WINNERS.items() if w['estado']==ESTADO_CONSTANTE]
SIN_DATOS=[k for k,w in WINNERS.items() if w['estado']==ESTADO_SIN_DATOS]
for k in SIN_DATOS: log('pronostico',f'{k[0]}/{k[1]}',ESTADO_SIN_DATOS,WINNERS[k]['motivo'])
for k in CONSTANTES: log('pronostico',f'{k[0]}/{k[1]}',ESTADO_CONSTANTE,'Se proyecta la mediana de los últimos 30 días (sin ML)')
print(f'Series: {len(ACTIVAS)} con modelo ML, {len(CONSTANTES)} constantes, {len(SIN_DATOS)} sin datos recientes (no se proyectan)')

def historia_recursion(m,var,hasta):
    hist,_=causal_fill(load_meter(m)[var].loc[:hasta]);return hist
_desde_global=min(WINNERS[k]['last'] for k in ACTIVAS+CONSTANTES)+pd.Timedelta(minutes=15)
# Calendario/exógenas de TODO el horizonte, vectorizado y una sola vez (no dependen de la recursión).
CAL_GLOBAL=features_calendario(pd.date_range(_desde_global,FIN_PLAN,freq='15min',name='Fecha'),
    FERIADOS,CLIMATOLOGIA_TEMPERATURA,PBI_MENSUAL,POBLACION_ANUAL)

# --- Validación en vivo contra features() del notebook: el calendario vectorizado y la
# recursión rápida deben reproducir EXACTAMENTE lo que daría features() paso a paso. ---
_k0=next((k for k in ACTIVAS if WINNERS[k]['name'] in ('Ridge','LightGBM','XGBoost','CatBoost')),ACTIVAS[0])
_w0=WINNERS[_k0];_hist0=historia_recursion(*_k0,_w0['last'])
_Xref=features(load_meter(_k0[0])[_k0[1]].tail(4000))
_cal_ref=features_calendario(_Xref.index,FERIADOS,CLIMATOLOGIA_TEMPERATURA,PBI_MENSUAL,POBLACION_ANUAL)
for _c in _cal_ref.columns:
    assert np.allclose(_cal_ref[_c].to_numpy(float),_Xref[_c].to_numpy(float),atol=1e-9,equal_nan=True),f'calendario difiere de features(): {_c}'
_m0=cargar_modelo(_w0['ruta_modelo']);preparar_para_inferencia(_m0)
_idx0=pd.date_range(_w0['last']+pd.Timedelta(minutes=15),periods=24,freq='15min')
_s=_hist0.tail(BUFFER_HISTORIA).copy();_ref=[]
for _t in _idx0:
    _s.loc[_t]=np.nan;_v=float(_m0.predict(features(_s).loc[[_t],_w0['columns']])[0]);_s.loc[_t]=_v;_ref.append(_v);_s=_s.tail(BUFFER_HISTORIA)
_rap=proyectar_recursivo(_m0,_hist0.to_numpy()[-BUFFER_HISTORIA:],_w0['columns'],CAL_GLOBAL.loc[_idx0[0]:_idx0[-1]])
assert np.allclose(_ref,_rap,rtol=1e-6,atol=1e-6),'La recursión rápida no reproduce features() del notebook'
print(f'Validación: calendario y recursión rápida reproducen features() ({_k0[0]}/{_k0[1]}, {_w0["name"]}, diferencia máx. {np.abs(np.array(_ref)-_rap).max():.2e})')
del _m0,_Xref,_cal_ref,_s

def construir_frame(k,pred_ml):
    m,var=k;w=WINNERS[k];hist=historia_recursion(m,var,w['last'])
    idx=pd.date_range(w['last']+pd.Timedelta(minutes=15),FIN_PLAN,freq='15min',name='Fecha')
    res=PRED[(m,var,w['name'])];residual=(res.real-res.pred).dropna();lo,hi=np.quantile(residual,[.05,.95])
    p_ml=pd.Series(pred_ml,index=idx);perfil=pd.Series(calendar_profile(hist,idx),index=idx)
    san=evaluar_sanidad(p_ml,hist,CFG['umbral_sanidad'])
    # Guarda de sanidad: una proyección marcada como REVISAR usa el perfil calendario (sin
    # entrenamiento) como plan B; nunca se recorta en silencio y ambas columnas quedan guardadas.
    usar_perfil=san['estado']=='REVISAR';fuente='PERFIL_CALENDARIO' if usar_perfil else 'ML'
    final=(perfil if usar_perfil else p_ml).to_numpy()
    return pd.DataFrame({'Fecha':idx,'pred':final,'banda_inf_exploratoria':final+lo,'banda_sup_exploratoria':final+hi,
        'pred_ml':p_ml.to_numpy(),'pred_perfil':perfil.to_numpy(),'fuente':pd.Categorical([fuente]*len(idx)),
        'estado_sanidad':pd.Categorical([san['estado']]*len(idx)),'marcas_sanidad':pd.Categorical([san['marcas']]*len(idx)),
        'desvio_media':san['desvio_media'],'ratio_std':san['ratio_std']})

def construir_frame_constante(k):
    m,var=k;w=WINNERS[k];hist=historia_recursion(m,var,w['last'])
    idx=pd.date_range(w['last']+pd.Timedelta(minutes=15),FIN_PLAN,freq='15min',name='Fecha')
    valor=float(hist.tail(96*30).median());c=np.full(len(idx),valor)
    return pd.DataFrame({'Fecha':idx,'pred':c,'banda_inf_exploratoria':c,'banda_sup_exploratoria':c,'pred_ml':c,'pred_perfil':c,
        'fuente':pd.Categorical([ESTADO_CONSTANTE]*len(idx)),'estado_sanidad':pd.Categorical(['NO_APLICA']*len(idx)),
        'marcas_sanidad':pd.Categorical(['']*len(idx)),'desvio_media':np.nan,'ratio_std':np.nan})

def registrar_proyeccion(k,frame):
    m,var=k;w=WINNERS[k];hist=historia_recursion(m,var,w['last']);FORECASTS[k]=frame
    SANIDAD.append(dict(medidor=m,variable=var,modelo=w['name'],fuente=str(frame['fuente'].iloc[0]),
        estado=str(frame['estado_sanidad'].iloc[0]),marcas=str(frame['marcas_sanidad'].iloc[0]),
        desvio_media=frame['desvio_media'].iloc[0],ratio_std=frame['ratio_std'].iloc[0]))
    frame_por_fecha=frame.set_index('Fecha')
    # Curva de proyección mensual: a resolución de 15 min un horizonte de años es
    # ilegible: se promedia por mes para ver claramente EN QUÉ MESES crece o decrece
    # la demanda proyectada, con el histórico reciente de contexto y la banda exploratoria.
    try:
        hist_mensual=hist.resample('MS').mean().tail(24)
        pred_mensual=frame_por_fecha['pred'].resample('MS').mean()
        banda_inf_mensual=frame_por_fecha['banda_inf_exploratoria'].resample('MS').mean()
        banda_sup_mensual=frame_por_fecha['banda_sup_exploratoria'].resample('MS').mean()
        # Se guarda para la figura COMBINADA por medidor (P+Q en un solo panel, mismo
        # formato que curva_{medidor}.png), generada despues de este bucle una vez que
        # ambas variables del medidor ya tienen su pronostico mensual calculado.
        MENSUAL[k]=dict(hist=hist_mensual,pred=pred_mensual,
            banda_inf=banda_inf_mensual,banda_sup=banda_sup_mensual,last=w['last'])
        # Variación mensual (%): identifica directamente los meses donde la demanda proyectada
        # CRECE (verde) frente a los meses donde, en parte, se consume MENOS que el mes previo
        # (rojo) -- y por construcción es consistente con la curva de arriba, no otra fuente.
        crecimiento=pred_mensual.pct_change()*100
        fig,ax=plt.subplots(figsize=(14,3))
        ax.bar(crecimiento.index,crecimiento.values,width=20,
               color=['tab:green' if v>=0 else 'tab:red' for v in crecimiento.fillna(0).values])
        ax.axhline(0,color='black',lw=.8)
        ax.set_ylabel('% cambio mensual');ax.set_title(f'{m} — {var}: variación mensual proyectada (verde=crece, rojo=decrece)')
        fig.tight_layout();fig.savefig(RUN/f'figuras/crecimiento_{TOKENS[m]}_{var}.png',dpi=120);plt.close(fig)
    except Exception as exc: log('figura_proyeccion',f'{m}/{var}','ERROR',exc)
    # Un archivo por año calendario; por la convención de fin de intervalo (celda 12) el
    # timestamp 00:00 del 1-ene pertenece al año que termina.
    for year in range(w['last'].year,FIN_PLAN.year+1):
        chunk=frame_por_fecha.loc[(frame_por_fecha.index>pd.Timestamp(year,1,1))&(frame_por_fecha.index<=pd.Timestamp(year+1,1,1))]
        if chunk.empty:continue
        chunk.reset_index().to_parquet(RUN/'pronosticos'/f'operativo_{TOKENS[m]}_{var}_{year}.parquet',index=False)
        if CFG['exportar_excel']:chunk[['pred','banda_inf_exploratoria','banda_sup_exploratoria']].reset_index().to_excel(
            RUN/'pronosticos'/f'operativo_{TOKENS[m]}_{var}_{year}.xlsx',engine='xlsxwriter')

def _ruta_serie(k): return RUN/'pronosticos'/f'serie_{TOKENS[k[0]]}_{k[1]}.parquet'
def _guardar_serie(k,frame): frame.to_parquet(_ruta_serie(k),index=False)
_pendientes=[]
for k in ACTIVAS+CONSTANTES:
    # Checkpoint por serie: al reanudar, lo ya proyectado no se recalcula.
    if CFG.get('reanudar_run') and _ruta_serie(k).exists(): registrar_proyeccion(k,pd.read_parquet(_ruta_serie(k)))
    elif k in CONSTANTES:
        frame=construir_frame_constante(k);_guardar_serie(k,frame);registrar_proyeccion(k,frame)
    else: _pendientes.append(k)
print(f'{len(FORECASTS)} series ya proyectadas (checkpoint/constantes); {len(_pendientes)} por proyectar con {CFG["workers_pronostico"]} procesos')

def _tareas():
    for k in _pendientes:
        m,var=k;w=WINNERS[k];hist=historia_recursion(m,var,w['last'])
        idx=pd.date_range(w['last']+pd.Timedelta(minutes=15),FIN_PLAN,freq='15min',name='Fecha')
        if len(idx)==0: log('pronostico',f'{m}/{var}','ERROR','Horizonte vacío (último dato >= fin_plan)');continue
        cols_cal=[c for c in w['columns'] if not es_columna_dinamica(c)]
        yield dict(clave=k,ruta_modelo=w['ruta_modelo'],columnas=w['columns'],historia=hist.to_numpy()[-BUFFER_HISTORIA:],
                   calendario_columnas=cols_cal,calendario_valores=CAL_GLOBAL.loc[idx[0]:idx[-1],cols_cal].to_numpy(float),
                   indice_futuro=idx)
_t0=time.time();_hechas=0
for k,pred,seg,error in proyectar_series(_tareas(),n_jobs=CFG['workers_pronostico']):
    _hechas+=1
    if error: log('pronostico',f'{k[0]}/{k[1]}','ERROR',error);print(f'[{_hechas}/{len(_pendientes)}] {k[0]}/{k[1]}: ERROR {error}',flush=True);continue
    try:
        frame=construir_frame(k,pred);_guardar_serie(k,frame);registrar_proyeccion(k,frame)
        print(f'[{_hechas}/{len(_pendientes)}] {k[0]}/{k[1]}: {len(frame)} pasos con {WINNERS[k]["name"]} en {seg:.0f} s '
              f'(sanidad {frame["estado_sanidad"].iloc[0]}{" "+str(frame["marcas_sanidad"].iloc[0]) if frame["marcas_sanidad"].iloc[0] else ""}); '
              f'transcurrido {(time.time()-_t0)/60:.1f} min',flush=True)
    except Exception as exc: log('pronostico',f'{k[0]}/{k[1]}','ERROR',exc);print(f'{k}: ERROR {exc}',flush=True)

SANIDAD_DF=pd.DataFrame(SANIDAD);SANIDAD_DF.to_csv(RUN/'tablas/sanidad_proyecciones.csv',index=False)
_estado_proy=pd.DataFrame([dict(medidor=m,variable=v,modelo_operativo=w['name'],estado_serie=w['estado'],motivo=w['motivo'],
    proyectada=(m,v) in FORECASTS,fuente=str(FORECASTS[(m,v)]['fuente'].iloc[0]) if (m,v) in FORECASTS else '')
    for (m,v),w in WINNERS.items()])
_estado_proy.to_csv(RUN/'tablas/estado_proyeccion.csv',index=False)
print(f'Proyecciones: {len(FORECASTS)} series; fuente: {_estado_proy.fuente.value_counts().to_dict()}; '
      f'sanidad REVISAR: {(SANIDAD_DF.estado=="REVISAR").sum() if len(SANIDAD_DF) else 0}')

# Curva de PROYECCION combinada por medidor: mismo formato que curva_{medidor}.png
# (panel unico, P=azul/Q=verde/S=morado superpuestos) -- el tramo observado en linea
# solida continua DIRECTAMENTE en linea punteada del mismo color para el pronostico,
# con una linea vertical marcando el ultimo dato observado. No existe un modelo ML
# que pronostique S directamente (solo P y Q se entrenan); S se DERIVA igual que
# S_calculada en la limpieza historica (celda 10): S=hypot(P,Q). Para el tramo
# observado se usa S_medida real cuando el medidor la trae (mismo dato que curva_);
# para el pronostico, unicamente hypot(P_pred,Q_pred), porque no existe S medida futura.
COLOR_VARIABLE={'P':'tab:blue','Q':'tab:green','S_medida':'tab:purple'}
ETIQUETA_VARIABLE={'P':'P','Q':'Q','S_medida':'S'}
_backend_prev=matplotlib.get_backend();plt.switch_backend('Agg')
for m in MEDIDORES:
    disponibles=[var for var in ('P','Q','S_medida') if (m,var) in MENSUAL]
    if not disponibles: continue
    try:
        fig,ax=plt.subplots(figsize=(14,5))
        for var in disponibles:
            bloque=MENSUAL[(m,var)];color=COLOR_VARIABLE[var];etiqueta=ETIQUETA_VARIABLE[var]
            bloque['hist'].plot(ax=ax,color=color,lw=1.3,label=f'{etiqueta} observado (prom. mensual)')
            bloque['pred'].plot(ax=ax,color=color,lw=1.3,ls='--',label=f'{etiqueta} pronóstico (prom. mensual, modelo propio)')
            ax.fill_between(bloque['pred'].index,bloque['banda_inf'],bloque['banda_sup'],color=color,alpha=.12)
        # Respaldo SOLO si el medidor no tiene KVA real entrenado (5/50): se sigue
        # derivando S=hypot(P,Q) para no perder la 3ra curva en esos casos.
        if 'S_medida' not in disponibles and 'P' in disponibles and 'Q' in disponibles:
            bp,bq=MENSUAL[(m,'P')],MENSUAL[(m,'Q')]
            d_m=load_meter(m)
            s_medida_mensual=(d_m['S_medida'].resample('MS').mean().tail(24)
                if 'S_medida' in d_m.columns and d_m['S_medida'].notna().any() else None)
            hist_S=s_medida_mensual if s_medida_mensual is not None else np.hypot(bp['hist'],bq['hist'])
            pred_S=pd.Series(np.hypot(bp['pred'],bq['pred']),index=bp['pred'].index)
            etiqueta_hist_s='S medido' if s_medida_mensual is not None else 'S calculado (hypot P,Q)'
            hist_S.plot(ax=ax,color='tab:purple',lw=1.3,label=f'{etiqueta_hist_s} (prom. mensual)')
            pred_S.plot(ax=ax,color='tab:purple',lw=1.3,ls='--',label='S pronóstico (derivado hypot P,Q -- sin KVA real, prom. mensual)')
        ultimo_observado=max(MENSUAL[(m,var)]['last'] for var in disponibles)
        ax.axvline(ultimo_observado,color='gray',ls=':',lw=.8)
        ax.set_ylabel('Potencia (kW / kvar / kVA)')
        ax.set_title(f'{m} — proyección mensual hasta {FIN_PLAN.date()} ({"/".join(ETIQUETA_VARIABLE[v] for v in disponibles)} continuando desde el último dato observado)')
        ax.legend(loc='upper left',fontsize=8)
        fig.tight_layout();fig.savefig(RUN/f'figuras/proyeccion_{TOKENS[m]}.png',dpi=120);plt.close(fig)
    except Exception as exc: log('figura_proyeccion_medidor',m,'ERROR',exc)
plt.close('all');plt.switch_backend(_backend_prev)
'''

CELDA_BACKTEST = r'''BACKTEST=[];H_MAX=max(CFG['backtest_horizontes'])
# Backtest SIN reentrenar: el modelo guardado se entrenó con el 80% cronológico más antiguo,
# así que los últimos test_days+1 días (dentro del tramo de prueba) son datos que nunca vio.
# Una sola recursión de H_MAX pasos sirve para todos los horizontes (la recursión es causal:
# los primeros h pasos son idénticos a una corrida de h pasos).
def _tareas_backtest():
    for k in ACTIVAS:
        m,var=k;w=WINNERS[k];cut=w['last']-pd.Timedelta(days=CFG['test_days']+1)
        idx=pd.date_range(cut+pd.Timedelta(minutes=15),periods=H_MAX,freq='15min',name='Fecha')
        cols_cal=[c for c in w['columns'] if not es_columna_dinamica(c)]
        cal=features_calendario(idx,FERIADOS,CLIMATOLOGIA_TEMPERATURA,PBI_MENSUAL,POBLACION_ANUAL)
        yield dict(clave=(m,var,cut),ruta_modelo=w['ruta_modelo'],columnas=w['columns'],
                   historia=historia_recursion(m,var,cut).to_numpy()[-BUFFER_HISTORIA:],
                   calendario_columnas=cols_cal,calendario_valores=cal[cols_cal].to_numpy(float),indice_futuro=idx)
for (m,var,cut),pred,seg,error in proyectar_series(_tareas_backtest(),n_jobs=CFG['workers_pronostico']):
    if error: log('backtest',f'{m}/{var}','ERROR',error);continue
    w=WINNERS[(m,var)];y=load_meter(m)[var]
    idx=pd.date_range(cut+pd.Timedelta(minutes=15),periods=H_MAX,freq='15min')
    fuera=bool(idx[0]>=w['test_inicio']) if w.get('test_inicio') is not None else None
    for h in CFG['backtest_horizontes']:
        BACKTEST.append(dict(medidor=m,variable=var,modelo=w['name'],horizonte=h,fuera_de_muestra=fuera,
                             **metrics(y.reindex(idx[:h]),pred[:h])))
pd.DataFrame(BACKTEST).to_csv(RUN/'tablas/backtest_recursivo.csv',index=False)
_bt=pd.DataFrame(BACKTEST)
if len(_bt): print(_bt.groupby('horizonte')[['RMSE','MAE','R2']].median().round(3).to_string())
'''

CELDA_DIGSILENT_2027 = r'''# --- 25b · Demandas proyectadas (máx/prom/mín) por medidor para el cierre del horizonte ---
# Mismo formato que la tabla de la sección 25, pero sobre la Matriz_Operativa proyectada (P y Q
# ya incluyen las altas de clientes confirmadas), para el último año completo del horizonte.
ANIO_PROY=(FIN_PLAN-pd.Timedelta(minutes=15)).year
_ini,_fin=pd.Timestamp(ANIO_PROY,1,1),pd.Timestamp(ANIO_PROY+1,1,1)   # fin de intervalo: 00:00 del 1-ene cierra el año
DEMANDAS_PROYECTADAS=[]
for m in MEDIDORES:
    _pc,_qc=f'{TOKENS[m]}_P',f'{TOKENS[m]}_Q'
    if _pc not in wide.columns or _qc not in wide.columns: continue
    _sub=wide.loc[(wide.index>_ini)&(wide.index<=_fin),[_pc,_qc]].dropna()
    if _sub.empty: continue
    _cobertura=len(_sub)/intervalos_esperados(ANIO_PROY);P,Q=_sub[_pc],_sub[_qc]
    _fuente=str(FORECASTS[(m,'P')]['fuente'].iloc[0]) if (m,'P') in FORECASTS else ''
    for _etq,_ts in (('Maxima_Demanda',P.idxmax()),('Minima_Demanda',P.idxmin())):
        _fila=punto_operacion(m,ANIO_PROY,P,Q,_ts,_etq,_cobertura);_fila['fuente_proyeccion']=_fuente;DEMANDAS_PROYECTADAS.append(_fila)
    _p,_q=float(P.mean()),float(Q.mean());_s=float(np.hypot(_p,_q))
    DEMANDAS_PROYECTADAS.append(dict(medidor=m,codigo=TOKENS[m],anio_base=ANIO_PROY,escenario='Demanda_Promedio',fecha=None,
        P_kW=_p,Q_kvar=_q,S_kVA=_s,FP=_p/_s if _s else np.nan,P_MW=_p/1000,Q_Mvar=_q/1000,S_MVA=_s/1000,
        cobertura_anio_base=round(_cobertura,4),cobertura_suficiente=bool(_cobertura>=UMBRAL_COBERTURA_ANIO_BASE),fuente_proyeccion=_fuente))
tabla_digsilent_proyectada=pd.DataFrame(DEMANDAS_PROYECTADAS)
if len(tabla_digsilent_proyectada):
    tabla_digsilent_proyectada.to_csv(RUN/f'tablas/Escenarios_DIgSILENT_{ANIO_PROY}_proyectado.csv',index=False)
    if CFG['exportar_excel']:
        tabla_digsilent_proyectada.to_excel(RUN/f'tablas/Escenarios_DIgSILENT_{ANIO_PROY}_proyectado.xlsx',
            sheet_name='PowerFactory_proyectado',engine='xlsxwriter')
    print(f'Tabla de demandas proyectadas {ANIO_PROY}: {tabla_digsilent_proyectada.medidor.nunique()} medidores en',RUN/'tablas')
else: log('digsilent_proyectado','tabla','ADVERTENCIA','Sin filas: la matriz operativa no cubre el año final del horizonte')
'''

MD_RESTAURAR = """## 14b · Restauración del estado (solo al reanudar)

Con `CFG['reanudar_run']` (CLI: `--reanudar <carpeta>`) las secciones 14–16 no se ejecutan: sus resultados ya están en la corrida indicada. Esta celda reconstruye `PRED`/`WINNERS` desde los `.joblib` y `metricas.csv`, **valida** que cada modelo guardado reproduzca su RMSE guardado y clasifica cada serie (`OK`, `SERIE_CONSTANTE`, `SIN_DATOS_RECIENTES`). El ganador operativo es el mejor modelo **ML** por `CV_RMSE`: los benchmarks `Estacional_*` quedan en las tablas, pero no se usan como pronóstico (recursivamente, `Estacional_1` es una línea plana)."""

MD_SEC20 = """## 20 · Pronóstico recursivo operativo

Aplica a P, Q y -- para los medidores con KVA real -- tambien a S_medida, con su propio modelo. **Sin reentrenar**: se usan los modelos guardados (entrenados con el 80% cronológico del histórico; `CFG['refit_final']` no existe a propósito). Recursivo, cada 15 min, con el modelo ML ganador, desde el último dato observado de cada serie hasta `CFG['fin_plan']` (cierre de 2027); un archivo por año. El cálculo usa `prodma.recursivo`: calendario/exógenas vectorizados una sola vez y features dinámicas con numpy sobre un buffer de 4 semanas (validado en vivo contra `features()`), con una serie por proceso y un checkpoint por serie. Las series sin datos recientes no se proyectan y las constantes se proyectan como constante declarada. **Guardas de sanidad**: cada proyección se compara con los últimos 12 meses; si se aparta más de `CFG['umbral_sanidad']`, se aplana o sale de rango, queda marcada y usa el perfil calendario (ambas columnas se guardan). Los intervalos son empíricos a un paso y no están validados a este horizonte. Por cada medidor/variable se grafica la curva de proyección mensual y la variación mensual (%)."""

MD_SEC21 = """## 21 · Backtesting multistep

Prueba recursiva sin reentrenar (modelo guardado) sobre los últimos `test_days`+1 días del tramo de prueba, con horizontes `CFG['backtest_horizontes']` (96 = 1 día, 672 = 1 semana). Se registra por separado del error rolling de un paso; `fuera_de_muestra` indica si la ventana quedó fuera del entrenamiento."""


def parte_1(nb) -> None:
    """Reanudación, horizonte 2027, pronóstico rápido, sanidad, backtest y verificación."""

    # 1) CFG: horizonte a fin de 2027 y parámetros de la reanudación
    i = celda_con(nb, "CFG = dict(")
    reemplazar(nb, i, "fin_plan='2037-01-01 00:00',",
               "fin_plan='2028-01-01 00:00',  # cierre de 2027 (fin de intervalo; 2037-01-01 00:00 era el cierre de 2036)")
    reemplazar(nb, i, "usar_pbi=True, usar_poblacion=True)",
               "usar_pbi=True, usar_poblacion=True,\n"
               "           # --- Cierre 2027 / reanudación sin reentrenar (ver docs/MANUAL_EJECUCION.md) ---\n"
               "           reanudar_run=None,        # ruta de una corrida previa: continúa ahí, sin entrenar\n"
               "           max_dias_sin_datos=180,   # series sin dato en más de N días respecto al último global: no se proyectan\n"
               "           umbral_sanidad=0.25,      # desvío máx. de la media proyectada vs. últimos 12 meses antes de marcar REVISAR\n"
               "           workers_pronostico=6,     # procesos para el pronóstico recursivo (cada uno carga su modelo: cuidar la RAM)\n"
               "           backtest_horizontes=(96,672), max_medidores_adicionales=None)")

    # 2) Carpeta de resultados: reutilizable al reanudar
    i = celda_con(nb, "RUN=ROOT/(f'{modo_tag}")
    fuente = nb.cells[i].source
    nb.cells[i].source = fuente[:fuente.index("RAIZ_PROYECTO=encontrar_raiz_proyecto()")] + CELDA_RUN

    # 3) Transformación del target: vive en el paquete (los .joblib no dependen de __main__)
    i = celda_con(nb, "def _transformar_y(v):")
    fuente = nb.cells[i].source
    a, b = fuente.index("def _transformar_y(v):"), fuente.index("def models():")
    nb.cells[i].source = (fuente[:a] + "from prodma.transformaciones import _transformar_y,_destransformar_y  # en el paquete: "
                          "los .joblib no dependen de __main__\n" + fuente[b:])

    # 4) Sección 14: partir la celda; el bucle de entrenamiento se omite al reanudar
    i = celda_con(nb, "def graficar_modelo_individual")
    lineas = nb.cells[i].source.split("\n")
    corte = next(k for k, l in enumerate(lineas) if l.startswith("for m in TARGETS:"))
    definiciones, bucle = "\n".join(lineas[:corte]).rstrip() + "\n", "\n".join(lineas[corte:])
    bucle = ("assert not CFG.get('reanudar_run'),'Reanudación activa: el entrenamiento no debe ejecutarse '\\\n"
             "    '(use la CLI: python -m prodma.pipeline --reanudar <carpeta>)'\n" + bucle)
    ancla_min = "      score,name,model=min(scores,key=lambda x:x[0])"
    if bucle.count(ancla_min) != 1:
        raise SystemExit("No se encontró la selección del ganador en la celda de entrenamiento")
    bucle = bucle.replace(ancla_min,
        "      # El ganador OPERATIVO es siempre un modelo ML: los benchmarks Estacional_* (modelo=int)\n"
        "      # se puntúan aparte, pero en recursión a meses/años 'Estacional_1' es una línea plana.\n"
        "      score,name,model=min((s for s in scores if not isinstance(s[2],(int,np.integer))),key=lambda x:x[0])")
    ancla_w = "WINNERS[(m,var)]={'name':name,'model':model,'cv_rmse':score,'last':last,'columns':columnas_ga}"
    if bucle.count(ancla_w) != 1:
        raise SystemExit("No se encontró la creación de WINNERS en la celda de entrenamiento")
    bucle = bucle.replace(ancla_w, "WINNERS[(m,var)]={'name':name,'model':model,'cv_rmse':score,'last':last,'columns':columnas_ga,\n"
                                   "                       'test_inicio':Xe.index[0]}")
    nb.cells[i].source = definiciones
    celda_bucle = nueva_codigo(bucle, OMITIR)
    celda_restaurar = nueva_codigo(CELDA_RESTAURAR, SOLO_REANUDAR)
    md_restaurar = nbformat.v4.new_markdown_cell(MD_RESTAURAR)
    md_restaurar.metadata.update(SOLO_REANUDAR)
    nb.cells[i + 1:i + 1] = [celda_bucle, md_restaurar, celda_restaurar]

    # 5) Secciones 15, selección estricta y 16 (ya persistidas): se omiten al reanudar
    i_diag = celda_con(nb, "from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white\nfrom statsmodels.stats.stattools import durbin_watson, jarque_bera")
    i_estricta = celda_con(nb, "# --- Seleccion estadistica estricta")
    i_ols = celda_con(nb, "# Única sección donde F, R² ajustado y la significancia t-Student")
    for j in (i_diag, i_estricta, i_ols):
        nb.cells[j].metadata.update(OMITIR)
    reemplazar(nb, i_estricta, "aprobados=candidatos[candidatos['cumple_minimos']==True]",
               "aprobados=candidatos[(candidatos['cumple_minimos']==True)&(~candidatos['modelo'].str.startswith('Estacional_'))]")

    # 6) Sección 17: figuras reanudables y sin agotar la memoria
    i = celda_con(nb, "for (m,var,name),d in PRED.items():\n    e=d.real-d.pred;fig,ax=plt.subplots(4,2")
    nb.cells[i].source = CELDA_FIGURAS

    # 7) Sección 20: pronóstico rápido, paralelo, con checkpoints y guardas de sanidad
    i = celda_con(nb, "def recursive(model,history,index,columns):")
    nb.cells[i].source = CELDA_PRONOSTICO
    i = celda_con(nb, "## 20 · Pronóstico recursivo operativo", "markdown")
    nb.cells[i].source = MD_SEC20

    # 8) Sección 21: backtest sin reentrenar
    i = celda_con(nb, "from sklearn.base import clone\nBACKTEST=[]")
    nb.cells[i].source = CELDA_BACKTEST
    i = celda_con(nb, "## 21 · Backtesting multistep", "markdown")
    nb.cells[i].source = MD_SEC21

    # 9) Sección 22: horizonte parametrizado (antes fijo en 2036) y perfil calendario en el paquete
    i = celda_con(nb, "def calendar_profile(s,index):")
    fuente = nb.cells[i].source
    a, b = fuente.index("def calendar_profile(s,index):"), fuente.index("def _calendar_profile_referencia")
    fuente = fuente[:a] + "from prodma.sanidad import calendar_profile  # perfil calendario vectorizado (en el paquete)\n" + fuente[b:]
    nb.cells[i].source = fuente
    reemplazar(nb, i, "for year in range(last.year,2037):", "for year in range(last.year,pd.Timestamp(CFG['fin_plan']).year+1):")
    reemplazar(nb, i, "log('planificacion','2036','PENDIENTE'", "log('planificacion',str(pd.Timestamp(CFG['fin_plan']).year-1),'PENDIENTE'")
    i = celda_con(nb, "## 22 · Planificación hasta 2036", "markdown")
    reemplazar(nb, i, "## 22 · Planificación hasta 2036", "## 22 · Planificación hasta el cierre del horizonte")

    # 10) Sección 24: verificación de las proyecciones y limitaciones declaradas
    i = celda_con(nb, "checks=[]\nfor p in (RUN/'pronosticos').glob('*.parquet'):")
    reemplazar(nb, i, "pd.DataFrame(checks).to_csv(RUN/'tablas/verificacion_salidas.csv',index=False)",
        "pd.DataFrame(checks).to_csv(RUN/'tablas/verificacion_salidas.csv',index=False)\n"
        "# Verificación de CADA proyección operativa: cubre exactamente hasta el cierre del horizonte, sin huecos,\n"
        "# finita y no plana (salvo las series constantes declaradas).\n"
        "VERIF=[]\n"
        "for (m,var),frame in FORECASTS.items():\n"
        "    f=frame.set_index('Fecha')['pred'];fuente=str(frame['fuente'].iloc[0])\n"
        "    VERIF.append(dict(medidor=m,variable=var,fuente=fuente,pasos=len(f),inicio=f.index.min(),fin=f.index.max(),\n"
        "        termina_en_fin_plan=bool(f.index.max()==FIN_PLAN),indice_unico=bool(f.index.is_unique),\n"
        "        paso_15min=bool((f.index.to_series().diff().dropna()==pd.Timedelta(minutes=15)).all()),\n"
        "        finitos=bool(np.isfinite(f.to_numpy()).all()),no_plana=bool(f.std()>1e-9) or fuente==ESTADO_CONSTANTE,\n"
        "        min_kW=float(f.min()),max_kW=float(f.max())))\n"
        "VERIF=pd.DataFrame(VERIF);VERIF.to_csv(RUN/'tablas/verificacion_proyecciones.csv',index=False)\n"
        "_criterios=['termina_en_fin_plan','indice_unico','paso_15min','finitos','no_plana']\n"
        "_incumplen=VERIF[~VERIF[_criterios].all(axis=1)] if len(VERIF) else VERIF\n"
        "if len(_incumplen): log('verificacion','proyecciones','ERROR',f'{len(_incumplen)} series incumplen: '+', '.join(_incumplen.medidor+'/'+_incumplen.variable))\n"
        "print(f'Proyecciones verificadas: {len(VERIF)-len(_incumplen)}/{len(VERIF)} cumplen todos los criterios')")
    reemplazar(nb, i, "# Cierre de los criterios mínimos de aceptación estadística",
        "limitaciones+= ['Pronóstico operativo hasta el cierre de '+str(FIN_PLAN.year-1)+' con los modelos guardados (80% cronológico del histórico, sin reentrenar con el 100%); '\n"
        "    'recursión a 15 min sin cota de error validada a ese horizonte (backtest solo a 1 y 7 días)',\n"
        "    f'{len(SIN_DATOS)} series sin datos recientes NO se proyectan y {len(CONSTANTES)} series constantes se proyectan como constante (ver estado_proyeccion.csv)',\n"
        "    f'{int((SANIDAD_DF.fuente==\"PERFIL_CALENDARIO\").sum()) if len(SANIDAD_DF) else 0} series usan el perfil calendario como plan B por las guardas de sanidad (ver sanidad_proyecciones.csv)']\n"
        "# Cierre de los criterios mínimos de aceptación estadística")
    reemplazar(nb, i, "resultados=str(RUN),limitaciones=limitaciones)",
        "reanudacion_desde=CFG.get('reanudar_run'),horizonte_fin=str(FIN_PLAN),\n"
        "    reentrenamientos_ml=0 if CFG.get('reanudar_run') else 'entrenados en la sección 14 de esta corrida',\n"
        "    series_por_estado={e:int(n) for e,n in pd.Series([w['estado'] for w in WINNERS.values()]).value_counts().items()},\n"
        "    proyecciones_verificadas=f'{len(VERIF)-len(_incumplen)}/{len(VERIF)}',\n"
        "    resultados=str(RUN),limitaciones=limitaciones)")

    # 11) Nueva sección 25b: demandas proyectadas para el cierre del horizonte
    i = celda_con(nb, "ANIO_BASE = None\nUMBRAL_COBERTURA_ANIO_BASE")
    nb.cells.insert(i + 1, nueva_codigo(CELDA_DIGSILENT_2027))

    # 12) Sección 26: solo medidores con serie activa; alcance configurable
    i = celda_con(nb, "MODELOS_ADICIONALES=[]\nif TARGETS:")
    reemplazar(nb, i, "MODELOS_ADICIONALES=[]\nif TARGETS:\n    for medidor in TARGETS:",
        "MODELOS_ADICIONALES=[]\n"
        "_con_datos=[m for m in TARGETS if any(WINNERS.get((m,v),{}).get('estado')==ESTADO_OK for v in ('P','Q','S_medida'))]\n"
        "MEDIDORES_ADICIONALES=_con_datos[:CFG['max_medidores_adicionales']] if CFG.get('max_medidores_adicionales') else _con_datos\n"
        "if MEDIDORES_ADICIONALES:\n    for medidor in MEDIDORES_ADICIONALES:")



MARCA_PARTE_2 = "# (reanudación) figura ya generada"


def parte_2(nb) -> None:
    """Al reanudar, no regenerar lo pesado que ya está en la carpeta de la corrida (figuras de las
    secciones 9-10, ~30 s c/u, y el Excel de la matriz consolidada, 146 MB)."""
    bucle_curvas = "for _i,m in enumerate(MEDIDORES,1):\n    print(f'Figura: {m} ({_i}/{len(MEDIDORES)})',flush=True)"
    i = celda_con(nb, bucle_curvas)
    reemplazar(nb, i, bucle_curvas,
               "for _i,m in enumerate(MEDIDORES,1):\n"
               "    if REANUDANDO and (RUN/f'figuras/curva_{TOKENS[m]}.png').exists(): continue  " + MARCA_PARTE_2 + "\n"
               "    print(f'Figura: {m} ({_i}/{len(MEDIDORES)})',flush=True)")
    export_xlsx = "if CFG['exportar_excel']:\n    assert len(matrix)<1048576 and len(matrix.columns)<16384\n    matrix.to_excel("
    reemplazar(nb, i, export_xlsx,
               "if CFG['exportar_excel'] and not (REANUDANDO and (RUN/'tablas/Matriz_Consolidada.xlsx').exists()):\n"
               "    assert len(matrix)<1048576 and len(matrix.columns)<16384\n    matrix.to_excel(")
    bucle_limpio = "    for var in variables_out:\n        n_rec=int(out[var+'_recortado'].sum())"
    i = celda_con(nb, bucle_limpio)
    reemplazar(nb, i, bucle_limpio,
               "    for var in variables_out:\n"
               "        if REANUDANDO and (RUN/f'figuras/limpio_{TOKENS[m]}_{var}.png').exists(): continue  " + MARCA_PARTE_2 + "\n"
               "        n_rec=int(out[var+'_recortado'].sum())")


CELDA_PRONOSTICO_V2 = r'''import matplotlib
from prodma.recursivo import (features_calendario,proyectar_series,proyectar_recursivo,
    preparar_para_inferencia,cargar_modelo,es_columna_dinamica,BUFFER_MINIMO)
from prodma.sanidad import evaluar_sanidad,calendar_profile,perfil_anclado
from prodma.reanudar import clasificar_serie,ESTADO_OK,ESTADO_CONSTANTE,ESTADO_SIN_DATOS
BUFFER_HISTORIA=BUFFER_MINIMO+96   # 4 semanas + 1 día: alcance máximo de las features dinámicas
FIN_PLAN=pd.Timestamp(CFG['fin_plan'])
H_SEL=max(CFG['backtest_horizontes'])   # horizonte (pasos de 15 min) con el que se elige el método de cada serie
FORECASTS={};MENSUAL={};SANIDAD=[]

# --- Estado de cada serie: solo las ACTIVAS entran a la elección de método. Las constantes se
# proyectan como constante declarada y las que dejaron de reportar NO se proyectan (proyectar
# años sobre un medidor apagado sería inventar demanda). ---
_ref_ultimo=max(w['last'] for w in WINNERS.values())
for (m,var),w in WINNERS.items():
    if 'estado' not in w: w['estado'],w['motivo']=clasificar_serie(w['cv_rmse'],w['last'],_ref_ultimo,CFG['max_dias_sin_datos'])
    if not w.get('ruta_modelo'): w['ruta_modelo']=str(RUN/'modelos'/f"{TOKENS[m]}_{var}_{w['name']}.joblib")
ACTIVAS=[k for k,w in WINNERS.items() if w['estado']==ESTADO_OK]
CONSTANTES=[k for k,w in WINNERS.items() if w['estado']==ESTADO_CONSTANTE]
SIN_DATOS=[k for k,w in WINNERS.items() if w['estado']==ESTADO_SIN_DATOS]
for k in SIN_DATOS: log('pronostico',f'{k[0]}/{k[1]}',ESTADO_SIN_DATOS,WINNERS[k]['motivo'])
for k in CONSTANTES: log('pronostico',f'{k[0]}/{k[1]}',ESTADO_CONSTANTE,'Se proyecta la mediana de los últimos 30 días (sin ML)')
print(f'Series: {len(ACTIVAS)} activas, {len(CONSTANTES)} constantes, {len(SIN_DATOS)} sin datos recientes (no se proyectan)')

def historia_recursion(m,var,hasta):
    hist,_=causal_fill(load_meter(m)[var].loc[:hasta]);return hist
def _perfiles(hist,idx):
    return {'PERFIL_ANCLADO':perfil_anclado(hist,idx),'PERFIL_CALENDARIO':calendar_profile(hist,idx)}
_desde_global=min(WINNERS[k]['last'] for k in ACTIVAS+CONSTANTES)+pd.Timedelta(minutes=15)
# Calendario/exógenas de TODO el horizonte, vectorizado y una sola vez (no dependen de la recursión).
CAL_GLOBAL=features_calendario(pd.date_range(_desde_global,FIN_PLAN,freq='15min',name='Fecha'),
    FERIADOS,CLIMATOLOGIA_TEMPERATURA,PBI_MENSUAL,POBLACION_ANUAL)

# --- Validación en vivo contra features() del notebook: el calendario vectorizado y la
# recursión rápida deben reproducir EXACTAMENTE lo que daría features() paso a paso. ---
_k0=next((k for k in ACTIVAS if WINNERS[k]['name'] in ('Ridge','LightGBM','XGBoost','CatBoost')),ACTIVAS[0])
_w0=WINNERS[_k0];_hist0=historia_recursion(*_k0,_w0['last'])
_Xref=features(load_meter(_k0[0])[_k0[1]].tail(4000))
_cal_ref=features_calendario(_Xref.index,FERIADOS,CLIMATOLOGIA_TEMPERATURA,PBI_MENSUAL,POBLACION_ANUAL)
for _c in _cal_ref.columns:
    assert np.allclose(_cal_ref[_c].to_numpy(float),_Xref[_c].to_numpy(float),atol=1e-9,equal_nan=True),f'calendario difiere de features(): {_c}'
_m0=cargar_modelo(_w0['ruta_modelo']);preparar_para_inferencia(_m0)
_idx0=pd.date_range(_w0['last']+pd.Timedelta(minutes=15),periods=24,freq='15min')
_s=_hist0.tail(BUFFER_HISTORIA).copy();_ref=[]
for _t in _idx0:
    _s.loc[_t]=np.nan;_v=float(_m0.predict(features(_s).loc[[_t],_w0['columns']])[0]);_s.loc[_t]=_v;_ref.append(_v);_s=_s.tail(BUFFER_HISTORIA)
_rap=proyectar_recursivo(_m0,_hist0.to_numpy()[-BUFFER_HISTORIA:],_w0['columns'],CAL_GLOBAL.loc[_idx0[0]:_idx0[-1]])
assert np.allclose(_ref,_rap,rtol=1e-6,atol=1e-6),'La recursión rápida no reproduce features() del notebook'
print(f'Validación: calendario y recursión rápida reproducen features() ({_k0[0]}/{_k0[1]}, {_w0["name"]}, diferencia máx. {np.abs(np.array(_ref)-_rap).max():.2e})')
del _m0,_Xref,_cal_ref,_s

# --- Backtest multi-paso SIN reentrenar y elección del método de cada serie -----------------------
# El pronóstico recursivo de un modelo entrenado a 1 paso pierde el nivel en pocos días (la
# recursión converge a un punto fijo); el R² a 1 paso no lo revela. Por eso el método de cada
# serie se elige por su error a H_SEL pasos (28 días) sobre los últimos test_days+1 días del tramo
# de prueba, que el modelo guardado nunca vio: ML recursivo, perfil calendario (mediana de 12
# meses) o perfil anclado (forma estacional x nivel reciente). El ML solo gana si mejora al mejor
# perfil en más de (1-margen_ml). SEMANAL (repetir la última semana) es solo referencia.
def _tareas_backtest():
    for k in ACTIVAS:
        m,var=k;w=WINNERS[k];cut=w['last']-pd.Timedelta(days=CFG['test_days']+1)
        idx=pd.date_range(cut+pd.Timedelta(minutes=15),periods=H_SEL,freq='15min',name='Fecha')
        cols_cal=[c for c in w['columns'] if not es_columna_dinamica(c)]
        cal=features_calendario(idx,FERIADOS,CLIMATOLOGIA_TEMPERATURA,PBI_MENSUAL,POBLACION_ANUAL)
        yield dict(clave=k,ruta_modelo=w['ruta_modelo'],columnas=w['columns'],
                   historia=historia_recursion(m,var,cut).to_numpy()[-BUFFER_HISTORIA:],
                   calendario_columnas=cols_cal,calendario_valores=cal[cols_cal].to_numpy(float),indice_futuro=idx)
_ruta_bt,_ruta_sel=RUN/'tablas/backtest_recursivo.csv',RUN/'tablas/seleccion_metodo.csv'
if CFG.get('reanudar_run') and _ruta_bt.exists() and _ruta_sel.exists() and 'metodo' in pd.read_csv(_ruta_bt,nrows=1).columns:
    BT_DF=pd.read_csv(_ruta_bt);SELECCION=pd.read_csv(_ruta_sel);print('Backtest y selección de método recuperados de la corrida previa')
else:
    _ml_bt={}
    for k,pred,seg,error in proyectar_series(_tareas_backtest(),n_jobs=CFG['workers_pronostico']):
        if error: log('backtest',f'{k[0]}/{k[1]}','ERROR',error)
        else: _ml_bt[k]=pred
    print(f'Backtest ML: {len(_ml_bt)}/{len(ACTIVAS)} series')
    BT=[];SEL=[]
    for k in ACTIVAS:
        m,var=k;w=WINNERS[k];cut=w['last']-pd.Timedelta(days=CFG['test_days']+1)
        idx=pd.date_range(cut+pd.Timedelta(minutes=15),periods=H_SEL,freq='15min')
        hist_bt=historia_recursion(m,var,cut);real=load_meter(m)[var].reindex(idx).to_numpy()
        cand=_perfiles(hist_bt,idx);cand['SEMANAL']=np.resize(hist_bt.tail(672).to_numpy(),H_SEL)
        if k in _ml_bt: cand['ML']=_ml_bt[k]
        fuera=bool(idx[0]>=w['test_inicio']) if w.get('test_inicio') is not None else None
        std_real=float(np.nanstd(real));fila=dict(medidor=m,variable=var,modelo_ml=w['name'],fuera_de_muestra=fuera,std_real=std_real)
        for nombre,p in cand.items():
            for h in CFG['backtest_horizontes']:
                BT.append(dict(medidor=m,variable=var,modelo_ml=w['name'],metodo=nombre,horizonte=h,fuera_de_muestra=fuera,
                               std_real=std_real,**metrics(real[:h],p[:h])))
            e=real-p;ok=np.isfinite(e)   # banda empírica (5%-95%) del error a H_SEL pasos: exploratoria
            if ok.sum()>=96: fila[f'lo_{nombre}'],fila[f'hi_{nombre}']=np.quantile(e[ok],[.05,.95])
            fila[f'rmse_{nombre}']=metrics(real,p)['RMSE']
        perfiles={n:fila.get(f'rmse_{n}',np.nan) for n in ('PERFIL_ANCLADO','PERFIL_CALENDARIO')}
        mejor_perfil=min(perfiles,key=lambda n:(np.isnan(perfiles[n]),perfiles[n]));rml=fila.get('rmse_ML',np.nan)
        fila['mejor_perfil']=mejor_perfil
        fila['metodo']='ML' if (np.isfinite(rml) and np.isfinite(perfiles[mejor_perfil]) and rml<CFG['margen_ml']*perfiles[mejor_perfil]) else mejor_perfil
        SEL.append(fila)
    BT_DF=pd.DataFrame(BT);SELECCION=pd.DataFrame(SEL)
    BT_DF.to_csv(_ruta_bt,index=False);SELECCION.to_csv(_ruta_sel,index=False)
SEL_IDX={(r['medidor'],r['variable']):r for r in SELECCION.to_dict('records')}
print('Método elegido por serie:',SELECCION.metodo.value_counts().to_dict())

def _banda(k,metodo):
    s=SEL_IDX[k];lo,hi=s.get(f'lo_{metodo}'),s.get(f'hi_{metodo}')
    return (0.0 if pd.isna(lo) else float(lo)),(0.0 if pd.isna(hi) else float(hi))
def construir_frame(k,pred_ml=None):
    m,var=k;w=WINNERS[k];s=SEL_IDX[k];hist=historia_recursion(m,var,w['last'])
    idx=pd.date_range(w['last']+pd.Timedelta(minutes=15),FIN_PLAN,freq='15min',name='Fecha')
    perf=_perfiles(hist,idx);fuente=s['metodo'];fallback=''
    if fuente=='ML':
        san_ml=evaluar_sanidad(pd.Series(pred_ml,index=idx),hist,CFG['umbral_sanidad'])
        if san_ml['estado']=='REVISAR':   # guarda de sanidad: nunca se recorta en silencio
            fallback='ML descartado por sanidad ('+san_ml['marcas']+')';fuente=s['mejor_perfil']
    final=np.asarray(pred_ml) if fuente=='ML' else perf[fuente]
    san=evaluar_sanidad(pd.Series(final,index=idx),hist,CFG['umbral_sanidad']);lo,hi=_banda(k,fuente)
    return pd.DataFrame({'Fecha':idx,'pred':final,'banda_inf_exploratoria':final+lo,'banda_sup_exploratoria':final+hi,
        'pred_ml':np.asarray(pred_ml) if pred_ml is not None else np.nan,
        'pred_perfil_anclado':perf['PERFIL_ANCLADO'],'pred_perfil_calendario':perf['PERFIL_CALENDARIO'],
        'fuente':pd.Categorical([fuente]*len(idx)),'estado_sanidad':pd.Categorical([san['estado']]*len(idx)),
        'marcas_sanidad':pd.Categorical([san['marcas']]*len(idx)),'fallback':pd.Categorical([fallback]*len(idx)),
        'desvio_media':san['desvio_media'],'ratio_std':san['ratio_std']})
def construir_frame_constante(k):
    m,var=k;w=WINNERS[k];hist=historia_recursion(m,var,w['last'])
    idx=pd.date_range(w['last']+pd.Timedelta(minutes=15),FIN_PLAN,freq='15min',name='Fecha')
    c=np.full(len(idx),float(hist.tail(96*30).median()))
    return pd.DataFrame({'Fecha':idx,'pred':c,'banda_inf_exploratoria':c,'banda_sup_exploratoria':c,'pred_ml':np.nan,
        'pred_perfil_anclado':c,'pred_perfil_calendario':c,'fuente':pd.Categorical([ESTADO_CONSTANTE]*len(idx)),
        'estado_sanidad':pd.Categorical(['NO_APLICA']*len(idx)),'marcas_sanidad':pd.Categorical(['']*len(idx)),
        'fallback':pd.Categorical(['']*len(idx)),'desvio_media':np.nan,'ratio_std':np.nan})

def registrar_proyeccion(k,frame):
    m,var=k;w=WINNERS[k];hist=historia_recursion(m,var,w['last']);FORECASTS[k]=frame
    SANIDAD.append(dict(medidor=m,variable=var,modelo_ml=w['name'],fuente=str(frame['fuente'].iloc[0]),
        estado=str(frame['estado_sanidad'].iloc[0]),marcas=str(frame['marcas_sanidad'].iloc[0]),
        fallback=str(frame['fallback'].iloc[0]),desvio_media=frame['desvio_media'].iloc[0],ratio_std=frame['ratio_std'].iloc[0]))
    frame_por_fecha=frame.set_index('Fecha')
    # Curva de proyección mensual: a resolución de 15 min un horizonte de años es
    # ilegible: se promedia por mes para ver claramente EN QUÉ MESES crece o decrece
    # la demanda proyectada, con el histórico reciente de contexto y la banda exploratoria.
    try:
        hist_mensual=hist.resample('MS').mean().tail(24)
        pred_mensual=frame_por_fecha['pred'].resample('MS').mean()
        banda_inf_mensual=frame_por_fecha['banda_inf_exploratoria'].resample('MS').mean()
        banda_sup_mensual=frame_por_fecha['banda_sup_exploratoria'].resample('MS').mean()
        # Se guarda para la figura COMBINADA por medidor (P+Q en un solo panel, mismo
        # formato que curva_{medidor}.png), generada despues de este bucle una vez que
        # ambas variables del medidor ya tienen su pronostico mensual calculado.
        MENSUAL[k]=dict(hist=hist_mensual,pred=pred_mensual,
            banda_inf=banda_inf_mensual,banda_sup=banda_sup_mensual,last=w['last'])
        # Variación mensual (%): identifica directamente los meses donde la demanda proyectada
        # CRECE (verde) frente a los meses donde, en parte, se consume MENOS que el mes previo
        # (rojo) -- y por construcción es consistente con la curva de arriba, no otra fuente.
        crecimiento=pred_mensual.pct_change()*100
        fig,ax=plt.subplots(figsize=(14,3))
        ax.bar(crecimiento.index,crecimiento.values,width=20,
               color=['tab:green' if v>=0 else 'tab:red' for v in crecimiento.fillna(0).values])
        ax.axhline(0,color='black',lw=.8)
        ax.set_ylabel('% cambio mensual');ax.set_title(f'{m} — {var}: variación mensual proyectada (verde=crece, rojo=decrece)')
        fig.tight_layout();fig.savefig(RUN/f'figuras/crecimiento_{TOKENS[m]}_{var}.png',dpi=120);plt.close(fig)
    except Exception as exc: log('figura_proyeccion',f'{m}/{var}','ERROR',exc)
    # Un archivo por año calendario; por la convención de fin de intervalo (celda 12) el
    # timestamp 00:00 del 1-ene pertenece al año que termina.
    for year in range(w['last'].year,FIN_PLAN.year+1):
        chunk=frame_por_fecha.loc[(frame_por_fecha.index>pd.Timestamp(year,1,1))&(frame_por_fecha.index<=pd.Timestamp(year+1,1,1))]
        if chunk.empty:continue
        chunk.reset_index().to_parquet(RUN/'pronosticos'/f'operativo_{TOKENS[m]}_{var}_{year}.parquet',index=False)
        if CFG['exportar_excel']:chunk[['pred','banda_inf_exploratoria','banda_sup_exploratoria']].reset_index().to_excel(
            RUN/'pronosticos'/f'operativo_{TOKENS[m]}_{var}_{year}.xlsx',engine='xlsxwriter')

def _ruta_serie(k): return RUN/'pronosticos'/f'serie_{TOKENS[k[0]]}_{k[1]}.parquet'
def _guardar_serie(k,frame): frame.to_parquet(_ruta_serie(k),index=False)
_pendientes_ml=[]
for k in ACTIVAS+CONSTANTES:
    # Checkpoint por serie: al reanudar, lo ya proyectado no se recalcula (borre pronosticos/serie_*.parquet
    # si cambia la lógica de proyección).
    if CFG.get('reanudar_run') and _ruta_serie(k).exists(): registrar_proyeccion(k,pd.read_parquet(_ruta_serie(k)))
    elif k in CONSTANTES:
        frame=construir_frame_constante(k);_guardar_serie(k,frame);registrar_proyeccion(k,frame)
    elif SEL_IDX[k]['metodo']=='ML': _pendientes_ml.append(k)   # el ML solo se proyecta a 2027 donde ganó el backtest
    else:
        frame=construir_frame(k);_guardar_serie(k,frame);registrar_proyeccion(k,frame)
print(f'{len(FORECASTS)} series proyectadas (perfiles/checkpoints/constantes); {len(_pendientes_ml)} por proyectar con ML ({CFG["workers_pronostico"]} procesos)')

def _tareas():
    for k in _pendientes_ml:
        m,var=k;w=WINNERS[k];hist=historia_recursion(m,var,w['last'])
        idx=pd.date_range(w['last']+pd.Timedelta(minutes=15),FIN_PLAN,freq='15min',name='Fecha')
        cols_cal=[c for c in w['columns'] if not es_columna_dinamica(c)]
        yield dict(clave=k,ruta_modelo=w['ruta_modelo'],columnas=w['columns'],historia=hist.to_numpy()[-BUFFER_HISTORIA:],
                   calendario_columnas=cols_cal,calendario_valores=CAL_GLOBAL.loc[idx[0]:idx[-1],cols_cal].to_numpy(float),
                   indice_futuro=idx)
_t0=time.time();_hechas=0
for k,pred,seg,error in proyectar_series(_tareas(),n_jobs=CFG['workers_pronostico']):
    _hechas+=1
    try:
        if error:   # el ML falló para esta serie: se usa el mejor perfil y queda registrado
            log('pronostico',f'{k[0]}/{k[1]}','ADVERTENCIA','ML falló ('+error+'); se usa '+SEL_IDX[k]['mejor_perfil'])
            SEL_IDX[k]['metodo']=SEL_IDX[k]['mejor_perfil'];frame=construir_frame(k)
        else: frame=construir_frame(k,pred)
        _guardar_serie(k,frame);registrar_proyeccion(k,frame)
        print(f'[{_hechas}/{len(_pendientes_ml)}] {k[0]}/{k[1]}: {len(frame)} pasos, fuente {frame["fuente"].iloc[0]} '
              f'({WINNERS[k]["name"]}, {seg:.0f} s); transcurrido {(time.time()-_t0)/60:.1f} min',flush=True)
    except Exception as exc: log('pronostico',f'{k[0]}/{k[1]}','ERROR',exc);print(f'{k}: ERROR {exc}',flush=True)

SANIDAD_DF=pd.DataFrame(SANIDAD);SANIDAD_DF.to_csv(RUN/'tablas/sanidad_proyecciones.csv',index=False)
_estado_proy=pd.DataFrame([dict(medidor=m,variable=v,modelo_ml=w['name'],estado_serie=w['estado'],motivo=w['motivo'],
    proyectada=(m,v) in FORECASTS,fuente=str(FORECASTS[(m,v)]['fuente'].iloc[0]) if (m,v) in FORECASTS else '')
    for (m,v),w in WINNERS.items()])
_estado_proy.to_csv(RUN/'tablas/estado_proyeccion.csv',index=False)
print(f'Proyecciones: {len(FORECASTS)} series; fuente: {_estado_proy.fuente.value_counts().to_dict()}; '
      f'sanidad REVISAR: {(SANIDAD_DF.estado=="REVISAR").sum() if len(SANIDAD_DF) else 0}')

# Curva de PROYECCION combinada por medidor: mismo formato que curva_{medidor}.png
# (panel unico, P=azul/Q=verde/S=morado superpuestos) -- el tramo observado en linea
# solida continua DIRECTAMENTE en linea punteada del mismo color para el pronostico,
# con una linea vertical marcando el ultimo dato observado. No existe un modelo ML
# que pronostique S directamente (solo P y Q se entrenan); S se DERIVA igual que
# S_calculada en la limpieza historica (celda 10): S=hypot(P,Q). Para el tramo
# observado se usa S_medida real cuando el medidor la trae (mismo dato que curva_);
# para el pronostico, unicamente hypot(P_pred,Q_pred), porque no existe S medida futura.
COLOR_VARIABLE={'P':'tab:blue','Q':'tab:green','S_medida':'tab:purple'}
ETIQUETA_VARIABLE={'P':'P','Q':'Q','S_medida':'S'}
_backend_prev=matplotlib.get_backend();plt.switch_backend('Agg')
for m in MEDIDORES:
    disponibles=[var for var in ('P','Q','S_medida') if (m,var) in MENSUAL]
    if not disponibles: continue
    try:
        fig,ax=plt.subplots(figsize=(14,5))
        for var in disponibles:
            bloque=MENSUAL[(m,var)];color=COLOR_VARIABLE[var];etiqueta=ETIQUETA_VARIABLE[var]
            bloque['hist'].plot(ax=ax,color=color,lw=1.3,label=f'{etiqueta} observado (prom. mensual)')
            bloque['pred'].plot(ax=ax,color=color,lw=1.3,ls='--',label=f'{etiqueta} proyección (prom. mensual, {FORECASTS[(m,var)]["fuente"].iloc[0]})')
            ax.fill_between(bloque['pred'].index,bloque['banda_inf'],bloque['banda_sup'],color=color,alpha=.12)
        # Respaldo SOLO si el medidor no tiene KVA real entrenado (5/50): se sigue
        # derivando S=hypot(P,Q) para no perder la 3ra curva en esos casos.
        if 'S_medida' not in disponibles and 'P' in disponibles and 'Q' in disponibles:
            bp,bq=MENSUAL[(m,'P')],MENSUAL[(m,'Q')]
            d_m=load_meter(m)
            s_medida_mensual=(d_m['S_medida'].resample('MS').mean().tail(24)
                if 'S_medida' in d_m.columns and d_m['S_medida'].notna().any() else None)
            hist_S=s_medida_mensual if s_medida_mensual is not None else np.hypot(bp['hist'],bq['hist'])
            pred_S=pd.Series(np.hypot(bp['pred'],bq['pred']),index=bp['pred'].index)
            etiqueta_hist_s='S medido' if s_medida_mensual is not None else 'S calculado (hypot P,Q)'
            hist_S.plot(ax=ax,color='tab:purple',lw=1.3,label=f'{etiqueta_hist_s} (prom. mensual)')
            pred_S.plot(ax=ax,color='tab:purple',lw=1.3,ls='--',label='S proyección (derivado hypot P,Q -- sin KVA real, prom. mensual)')
        ultimo_observado=max(MENSUAL[(m,var)]['last'] for var in disponibles)
        ax.axvline(ultimo_observado,color='gray',ls=':',lw=.8)
        ax.set_ylabel('Potencia (kW / kvar / kVA)')
        ax.set_title(f'{m} — proyección mensual hasta {FIN_PLAN.date()} ({"/".join(ETIQUETA_VARIABLE[v] for v in disponibles)} continuando desde el último dato observado)')
        ax.legend(loc='upper left',fontsize=8)
        fig.tight_layout();fig.savefig(RUN/f'figuras/proyeccion_{TOKENS[m]}.png',dpi=120);plt.close(fig)
    except Exception as exc: log('figura_proyeccion_medidor',m,'ERROR',exc)
plt.close('all');plt.switch_backend(_backend_prev)
'''

CELDA_BACKTEST_V2 = r'''# Resumen del backtest multi-paso (calculado en la sección 20, donde se usa para elegir el método de
# cada serie). RMSE relativo = RMSE / desvío estándar real de la ventana: 1.0 equivale a no saber nada
# más que la media; menor es mejor. Sin reentrenar: el modelo guardado nunca vio esos días
# (`fuera_de_muestra` lo indica por serie).
_bt=BT_DF.copy();_bt['rmse_rel']=_bt.RMSE/_bt.std_real.replace(0,np.nan)
print('RMSE relativo al desvío real (mediana entre series) por horizonte (pasos de 15 min) y método:')
print(_bt.pivot_table(index='horizonte',columns='metodo',values='rmse_rel',aggfunc='median').round(3).to_string())
_h=_bt[_bt.horizonte==H_SEL].pivot_table(index=['medidor','variable'],columns='metodo',values='RMSE')
print(f'\nVictorias por método a {H_SEL} pasos entre ML/PERFIL_ANCLADO/PERFIL_CALENDARIO:',
      _h[[c for c in ('ML','PERFIL_ANCLADO','PERFIL_CALENDARIO') if c in _h]].idxmin(axis=1).value_counts().to_dict())
print('Método elegido por serie (el ML solo si mejora al mejor perfil en más de',f"{1-CFG['margen_ml']:.0%}):",SELECCION.metodo.value_counts().to_dict())
if 'fuera_de_muestra' in SELECCION: print('Series con la ventana de backtest fuera del entrenamiento:',int(SELECCION.fuera_de_muestra.fillna(False).sum()),'de',len(SELECCION))
'''

MD_SEC20_V2 = """## 20 · Proyección operativa hasta el cierre del horizonte

Aplica a P, Q y -- para los medidores con KVA real -- también a S_medida. **Sin reentrenar**: se usan los modelos guardados (entrenados con el 80% cronológico del histórico). El pronóstico recursivo de un modelo entrenado a 1 paso **pierde el nivel en pocos días** (la recursión converge a un punto fijo; el R² a 1 paso no lo revela), así que el método de cada serie se **elige por backtest a 28 días** sobre datos que el modelo no vio: ML recursivo, perfil calendario (mediana de 12 meses) o perfil anclado (forma estacional × nivel reciente desestacionalizado). El ML solo se usa donde mejora al mejor perfil en más de `1-CFG['margen_ml']` y, además, pasa las guardas de sanidad (media proyectada frente a los últimos 12 meses, aplanamiento, rango); si no, se usa el mejor perfil y queda registrado. Cada serie deja `fuente`, `pred_ml`, `pred_perfil_anclado` y `pred_perfil_calendario` en `pronosticos/serie_*.parquet`. Las series sin datos recientes no se proyectan y las constantes se proyectan como constante declarada. El cálculo recursivo usa `prodma.recursivo` (calendario vectorizado + numpy, validado en vivo contra `features()`), un proceso por serie y un checkpoint por serie. Las bandas son empíricas del backtest a 28 días y exploratorias: no hay validación a 16 meses. Por cada medidor/variable se grafica la curva mensual y la variación mensual (%)."""

MD_SEC21_V2 = """## 21 · Backtesting multistep

Resumen del backtest recursivo **sin reentrenar** (se calcula en la sección 20 para poder elegir el método de cada serie): últimos `test_days`+1 días del tramo de prueba, horizontes `CFG['backtest_horizontes']` (96 = 1 día, 672 = 1 semana, 2688 = 28 días), para el ML y para los perfiles. Se registra por separado del error rolling de un paso; `fuera_de_muestra` indica si la ventana quedó fuera del entrenamiento."""


def parte_3(nb) -> None:
    """La recursión ML pierde el nivel en pocos días: el método de cada serie se elige por backtest
    (ML / perfil anclado / perfil calendario) en vez de asumir que el ML sirve a 16 meses."""
    i = celda_con(nb, "backtest_horizontes=(96,672), max_medidores_adicionales=None)")
    reemplazar(nb, i, "backtest_horizontes=(96,672), max_medidores_adicionales=None)",
               "backtest_horizontes=(96,672,2688),  # el mayor decide el método de cada serie\n"
               "           margen_ml=0.95,           # el ML solo se usa si su RMSE a 28 días es < margen x el del mejor perfil\n"
               "           max_medidores_adicionales=None)")
    i = celda_con(nb, "def construir_frame(k,pred_ml):")
    nb.cells[i].source = CELDA_PRONOSTICO_V2
    i = celda_con(nb, "BACKTEST=[];H_MAX=max(CFG['backtest_horizontes'])")
    nb.cells[i].source = CELDA_BACKTEST_V2
    i = celda_con(nb, "## 20 · Pronóstico recursivo operativo", "markdown")
    nb.cells[i].source = MD_SEC20_V2
    i = celda_con(nb, "## 21 · Backtesting multistep", "markdown")
    nb.cells[i].source = MD_SEC21_V2
    i = celda_con(nb, "limitaciones+= ['Pronóstico operativo hasta el cierre de '")
    fuente = nb.cells[i].source
    a = fuente.index("limitaciones+= [")
    b = fuente.index("# Cierre de los criterios mínimos de aceptación estadística")
    nb.cells[i].source = fuente[:a] + (
        "limitaciones+= ['Proyección hasta el cierre de '+str(FIN_PLAN.year-1)+' con los modelos guardados (80% cronológico del histórico, sin reentrenar con el 100%). '\n"
        "    'El pronóstico recursivo ML pierde el nivel en pocos días: el método de cada serie se eligió por backtest a 28 días (ML, perfil anclado o perfil calendario); '\n"
        "    'no hay validación a 16 meses y ninguna proyección incluye crecimiento más allá del nivel reciente',\n"
        "    f'{len(SIN_DATOS)} series sin datos recientes NO se proyectan y {len(CONSTANTES)} series constantes se proyectan como constante (ver estado_proyeccion.csv)',\n"
        "    f'Método de proyección por serie (ver seleccion_metodo.csv y sanidad_proyecciones.csv): {SANIDAD_DF.fuente.value_counts().to_dict() if len(SANIDAD_DF) else {}}']\n") + fuente[b:]
    reemplazar(nb, i, "    proyecciones_verificadas=f'{len(VERIF)-len(_incumplen)}/{len(VERIF)}',",
               "    proyecciones_verificadas=f'{len(VERIF)-len(_incumplen)}/{len(VERIF)}',\n"
               "    metodo_por_serie={e:int(n) for e,n in SANIDAD_DF.fuente.value_counts().items()} if len(SANIDAD_DF) else {},")


def parte_4(nb) -> None:
    """Detalles al reanudar: el registro de eventos no duplica lo ya registrado y RESUMEN.json va en UTF-8."""
    i = celda_con(nb, "LOG=pd.read_csv(RUN/'logs/eventos.csv').fillna('').to_dict('records')")
    reemplazar(nb, i,
               "    LOG=pd.read_csv(RUN/'logs/eventos.csv').fillna('').to_dict('records')  # conserva los eventos previos\n"
               "def log(stage,item,status,detail=''):\n"
               "    LOG.append(dict(etapa=stage,item=str(item),estado=status,detalle=str(detail)))\n",
               "    LOG=pd.read_csv(RUN/'logs/eventos.csv',dtype=str,keep_default_na=False).to_dict('records')  # conserva los eventos previos\n"
               "    LOG=[e for k,e in enumerate(LOG) if e not in LOG[:k]]   # sin duplicados de reanudaciones anteriores\n"
               "def log(stage,item,status,detail=''):\n"
               "    evento=dict(etapa=stage,item=str(item),estado=status,detalle=str(detail))\n"
               "    if REANUDANDO and evento in LOG: return   # una reanudación no repite eventos ya registrados\n"
               "    LOG.append(evento)\n")
    i = celda_con(nb, "(RUN/'RESUMEN.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))")
    reemplazar(nb, i, "(RUN/'RESUMEN.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))",
               "(RUN/'RESUMEN.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')")


def main() -> int:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    fuentes = [c.source for c in nb.cells if c.cell_type == "code"]
    cambios = False
    if not any("prodma.reanudar" in f for f in fuentes):
        parte_1(nb)
        cambios = True
    if not any(MARCA_PARTE_2 in f for f in fuentes):
        parte_2(nb)
        cambios = True
    if not any("H_SEL=max(" in f for f in fuentes):
        parte_3(nb)
        cambios = True
    if not any("sin duplicados de reanudaciones anteriores" in f for f in fuentes):
        parte_4(nb)
        cambios = True
    if not cambios:
        print("El notebook ya tiene los cambios del cierre 2027; no se modifica.")
        return 0
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)
    print("Notebook actualizado:", NOTEBOOK)
    return 0


if __name__ == "__main__":
    sys.exit(main())
