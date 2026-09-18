import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebook" / "proyecciondemanda.ipynb"


class NotebookVSCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.sources = ["".join(cell.get("source", [])) for cell in cls.notebook["cells"]]
        cls.full_source = "\n".join(cls.sources)

    def test_required_local_inputs_exist(self):
        expected = [
            *(PROJECT_ROOT / "datahistorica" / f"demanda{year}.xlsx" for year in range(2023, 2027)),
            PROJECT_ROOT / "nuevosclientes" / "nuevosclientes.xlsx",
        ]
        self.assertEqual([], [str(path) for path in expected if not path.is_file()])

    def test_notebook_no_longer_depends_on_colab_or_drive_api(self):
        forbidden = (
            "from google.colab",
            "googleapiclient",
            "drive.mount(",
            "api.files().list",
            "MediaIoBaseDownload",
            "/content/",
        )
        for token in forbidden:
            self.assertNotIn(token, self.full_source)

    def test_notebook_declares_portable_local_directories(self):
        config_source = self.sources[7]
        for token in (
            "encontrar_raiz_proyecto",
            "datahistorica",
            "nuevosclientes",
            "resultados",
        ):
            self.assertIn(token, config_source)

    def test_inventory_uses_all_four_historical_workbooks_and_clients(self):
        inventory_source = self.sources[9]
        self.assertIn("HIST_BASES", inventory_source)
        self.assertIn("DIRECTORIO_HISTORICO", inventory_source)
        self.assertIn("DIRECTORIO_CLIENTES", inventory_source)
        self.assertIn(".rglob(", inventory_source)

    def test_download_cell_is_replaced_by_local_file_validation(self):
        validation_source = self.sources[11]
        self.assertIn("resolve()", validation_source)
        self.assertIn("is_file()", validation_source)
        self.assertNotIn("next_chunk", validation_source)

    def test_stale_colab_outputs_and_metadata_are_removed(self):
        self.assertNotIn("colab", self.notebook.get("metadata", {}))
        for cell in self.notebook["cells"]:
            self.assertNotIn("colab", cell.get("metadata", {}))
            if cell.get("cell_type") == "code":
                self.assertEqual([], cell.get("outputs", []))
                self.assertIsNone(cell.get("execution_count"))

    def test_extended_forecasters_are_connected_to_notebook(self):
        self.assertIn("prodma.forecast_models", self.full_source)
        self.assertIn("comparar_modelos_adicionales", self.full_source)
        for name in ("ARIMA", "SARIMA", "HoltWinters", "Prophet", "LSTM"):
            self.assertIn(name, self.full_source)
        self.assertIn("probar_backends_gpu", self.full_source)


if __name__ == "__main__":
    unittest.main()
