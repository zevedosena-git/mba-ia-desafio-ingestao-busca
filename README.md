# Desafio MBA Engenharia de Software com IA - Full Cycle

Solução de RAG (Retrieval Augmented Generation) usando PGVector, LangChain e Google Gemini API.

## Pré-requisitos

- Python 3.8+
- Docker e Docker Compose
- Conta Google AI Studio com API Key

## Passo a Passo para Executar

### 1. Configurar variáveis de ambiente

crie um arquivo `.env` na raiz do projeto com:

```env
GOOGLE_API_KEY=sua_chave_api_aqui
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
GOOGLE_LLM_MODEL=gemini-2.5-flash-lite
PGVECTOR_URL=postgresql://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=doc_exerc
```

**Importante:** Substitua `sua_chave_api_aqui` pela sua chave real do Google AI Studio.

### 2. Instalar dependências Python

**Execute da raiz do projeto:**

```bash
pip install -r requirements.txt
```

Ou se estiver usando ambiente virtual:

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Windows CMD
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Iniciar o banco de dados PostgreSQL com PGVector

**Execute da raiz do projeto:**

```bash
docker-compose up -d
```

Aguarde alguns segundos para o banco inicializar e a extensão `vector` ser criada.

### 4. Verificar se o arquivo document.pdf está na raiz

**Importante:** O arquivo `document.pdf` deve estar na raiz do projeto (mesmo nível que `src/`, `docker-compose.yml`), pois o código usa caminho fixo relativo à raiz.

Estrutura esperada:
```
exercicioPGvector/
├── document.pdf          ← Deve estar aqui
├── src/
│   ├── ingest.py
│   ├── chat.py
│   └── search.py
├── docker-compose.yml
├── .env
└── requirements.txt
```

### 5. Executar o chat (fazer perguntas sobre o documento)

**Execute da raiz do projeto:**

```bash
python src/chat.py
```

Será validados se as configurações foram preenchidas.

Inicialmente verifica que o ingestion foi feito, caso contrário irá executar o processo antes de liberar o uso do chat.

**Nota:** Se encontrar erro 429 (quota excedida), o script aguarda 60 segundos e tenta novamente automaticamente.

O chat irá:
- Buscar trechos relevantes do documento usando busca semântica
- Usar o LLM Gemini para responder baseado apenas no contexto encontrado
- Respeitar as regras: só responde com base no documento, não inventa informações

## Estrutura do Projeto

- `src/ingest.py` - Processa o PDF e popula o banco com embeddings
- `src/chat.py` - Interface de chat RAG que responde perguntas sobre o documento
- `src/search.py` - Template de prompt e formatação de contexto
- `docker-compose.yml` - Configuração do PostgreSQL com PGVector
- `.env` - Variáveis de ambiente (não versionado)

## Observações Importantes

⚠️ **Todos os comandos Python devem ser executados da raiz do projeto**, pois o código usa caminho fixo `Path(__file__).resolve().parent.parent` para localizar o `document.pdf` na raiz.

Se executar de outro diretório, o arquivo não será encontrado e ocorrerá erro.