"""Leitura das tabelas Gold (Parquet) do Volume e ajuste dos nomes de colunas ao app.

Ordem de leitura:
1. Cópia local (pasta existe no disco)               -> lê os arquivos direto
2. Streamlit Cloud (DATABRICKS_HTTP_PATH nos Secrets) -> SQL Warehouse, 1 consulta por tabela
3. Databricks Apps                                    -> Files API com o service principal do app
"""
import io
import os

import pandas as pd
import streamlit as st

from . import config as C


class TabelaNaoEncontrada(Exception):
    pass


# ---------------- credenciais ----------------
def _segredo(nome):
    """Lê do st.secrets (Streamlit Cloud) ou da variável de ambiente (Databricks Apps)."""
    try:
        if nome in st.secrets:
            return str(st.secrets[nome]).strip()
    except Exception:  # sem secrets.toml
        pass
    v = os.getenv(nome)
    return v.strip() if v else None


def usa_warehouse() -> bool:
    return bool(_segredo("DATABRICKS_HTTP_PATH") and _segredo("DATABRICKS_TOKEN"))


# ---------------- 1) leitura local ----------------
def _ler_local(base, tabela):
    for alvo in (os.path.join(base, tabela), os.path.join(base, f"{tabela}.parquet")):
        if os.path.isdir(alvo):
            arqs = sorted(os.path.join(r, f) for r, _, fs in os.walk(alvo) for f in fs
                          if not f.startswith(("_", ".")) and f.endswith((".parquet", ".csv")))
            if arqs:
                return pd.concat([pd.read_csv(a) if a.endswith(".csv") else pd.read_parquet(a) for a in arqs],
                                 ignore_index=True)
        elif os.path.isfile(alvo):
            return pd.read_parquet(alvo)
    raise TabelaNaoEncontrada(f"{tabela} não encontrada em {base}")


# ---------------- 2) SQL Warehouse ----------------
def ler_sql(caminho: str, colunas: str = "*") -> pd.DataFrame:
    """Lê uma pasta Parquet do Volume via SQL Warehouse (uma consulta só)."""
    from databricks import sql
    host = _segredo("DATABRICKS_HOST").replace("https://", "").replace("http://", "").rstrip("/")
    with sql.connect(server_hostname=host,
                     http_path=_segredo("DATABRICKS_HTTP_PATH"),
                     access_token=_segredo("DATABRICKS_TOKEN"),
                     _socket_timeout=120) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {colunas} FROM parquet.`{caminho}`")
        return cur.fetchall_arrow().to_pandas()


# ---------------- 3) Files API (Databricks Apps) ----------------
@st.cache_resource
def _cliente():
    from databricks.sdk import WorkspaceClient
    host, token = _segredo("DATABRICKS_HOST"), _segredo("DATABRICKS_TOKEN")
    if host and token:
        if not host.startswith("http"):
            host = "https://" + host
        return WorkspaceClient(host=host.rstrip("/"), token=token)
    return WorkspaceClient()                 # Databricks Apps: credencial automática do app


def _listar_parquet(w, pasta):
    arqs = []
    for e in w.files.list_directory_contents(pasta):
        nome = e.name.rstrip("/")
        if nome.startswith(("_", ".")):
            continue
        if e.is_directory:
            arqs += _listar_parquet(w, e.path)
        elif nome.endswith(".parquet"):
            arqs.append(e.path)
    return arqs


def _ler_sdk(base, tabela):
    w = _cliente()
    entradas = {e.name.rstrip("/"): e for e in w.files.list_directory_contents(base)}
    for nome in (tabela, f"{tabela}.parquet"):
        e = entradas.get(nome)
        if e is None:
            continue
        arqs = _listar_parquet(w, e.path) if e.is_directory else [e.path]
        if arqs:
            return pd.concat([pd.read_parquet(io.BytesIO(w.files.download(a).contents.read())) for a in arqs],
                             ignore_index=True)
    raise TabelaNaoEncontrada(f"{tabela} não encontrada em {base}")


@st.cache_data(ttl=3600, show_spinner="Carregando dados da camada Gold...")
def ler_bruto(tabela):
    base = (_segredo("RAIOX_PATH") or C.BASE_PATH).rstrip("/")
    if os.path.isdir(base):                  # 1) cópia local
        return _ler_local(base, tabela)
    if usa_warehouse():                      # 2) Streamlit Cloud: SQL Warehouse
        return ler_sql(f"{base}/{tabela}")
    return _ler_sdk(base, tabela)            # 3) Databricks Apps


# ---------------- padronização ----------------
def mapear_colunas(tabela, colunas):
    """{nome_real: nome_do_app}, sem diferenciar maiúsculas."""
    reais = {str(c).lower(): c for c in colunas}
    return {reais[r.lower()]: l for l, r in C.COLUNAS.get(tabela, {}).items() if r.lower() in reais}


def padronizar(tabela, df):
    df = df.rename(columns=mapear_colunas(tabela, df.columns)).copy()
    for col in ("categoria", "uf_origem", "uf_destino"):
        if col in df.columns:
            s = df[col].where(df[col].notna()).astype("string").str.strip()
            df[col] = s.where(~s.str.lower().isin(C.CATEGORIAS_IGNORADAS)) if col == "categoria" else s
    if "mes" in df.columns:
        df["mes"] = pd.to_datetime(df["mes"], errors="coerce")
    for col in df.columns:
        if col not in ("categoria", "uf_origem", "uf_destino", "mes", "reclamacao_top"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in C.PERCENTUAIS & set(df.columns):
        df[col] = df[col] / 100
    return df


def carregar(tabela):
    return padronizar(tabela, ler_bruto(tabela))


def diagnostico():
    linhas = []
    for tabela, mapa in C.COLUNAS.items():
        try:
            df = ler_bruto(tabela)
            achadas = set(mapear_colunas(tabela, df.columns).values())
            faltando = [f"{l} → {r}" for l, r in mapa.items() if l not in achadas]
            linhas.append({"tabela": tabela, "status": "OK" if not faltando else "colunas faltando",
                           "linhas": len(df), "faltando": ", ".join(faltando) or "—",
                           "colunas": ", ".join(map(str, df.columns))})
        except Exception as e:  # noqa: BLE001
            linhas.append({"tabela": tabela, "status": "não encontrada", "linhas": 0,
                           "faltando": str(e)[:160], "colunas": "—"})
    return linhas
