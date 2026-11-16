"""Os testes do legado e do v2 exercitam esses caminhos: sem isto, o v3 carregado na importação responderia por eles."""
import pytest

from mapi_ai import app as app_module


@pytest.fixture(autouse=True)
def _sem_v3_por_padrao(monkeypatch):
    monkeypatch.setattr(app_module, "v3_model", None)
