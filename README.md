## Depression_Annotator_jury_LLMs_Laerty

Notebook para **classificação de severidade de depressão** (0–3) em textos do Reddit, usando **múltiplos LLMs via OpenRouter** e **DSPy** para compilação de programas (prompt/program search).

### O que este notebook faz

- **Configura 7 modelos no OpenRouter** com `dspy` + `litellm`:
  - Premium: `gpt-4o-mini`, `claude-3.5-haiku`, `qwen3-30b`, `mimo-v2-flash`.
  - Free: `nemotron-nano-12b-v2-vl`, `deepseek-r1-0528`, `arcee-trinity-large-preview`.
- Implementa um wrapper (`NemotronPercentCleaningLM`) para limpar `%` do JSON do Nemotron.
- Define a **Signature DSPy** `ClassifyDepression` com:
  - `text` (entrada),
  - `probability` (distribuição em JSON 0–1),
  - `rationale` (raciocínio clínico em 1 sentença),
  - `severity` (rótulo 0, 1, 2 ou 3).
- Cria um **predictor Chain-of-Thought** para classificar textos.

### Dados e baseline

- Usa o arquivo **`gold_standard_depseverity.csv`** com textos e rótulos `severity` (0–3).
- Seleciona 20 amostras com melhor desempenho zero-shot do **GPT-4o-mini** e registra em `zero_shot_4omini`.
- Calcula métricas de baseline (sem few-shot/compilação):
  - Matriz de confusão, Accuracy, Precision/Recall/F1 (macro/weighted),
  - **Quadratic Weighted Kappa (QWK)** para avaliar coerência ordinal.

### DSPy: treino, dev e métricas internas

- Separa:
  - `restantes_df` → **train_set** (exemplos DSPy com `text` e `severity`),
  - `gold_df_teste` → **dev_set**.
- Define duas métricas para o otimizador:
  - `metric_dist_penalty`: penaliza erros mais distantes (ordinais).
  - `classification_metric`: accuracy simples (0/1) sobre `severity`.
- Configura `Evaluate` com `dev_set` e `classification_metric`.

### Compilação dos modelos do júri

- Para cada modelo (`gpt4o_mini`, `claude_haiku`, `qwen3_30b`, `mimo_v2_flash`, `nemotron_12b`, `deepseek_r1`, `arcee_trinity`), o notebook:
  - Executa **BFRS** (Best-First-Region Search) do DSPy,
  - Amostra programas, avalia no `dev_set`,
  - Mostra evolução da métrica (ex.: 50%, 60%, 75%),
  - Registra melhor seed e confirma a compilação do modelo.

### Como reproduzir

1. Ter Python 3.10+ e instalar dependências (p. ex.: `dspy-ai`, `litellm`, `pandas`, `scikit-learn`, `jupyter`).
2. Criar `secrets.json` com `OPENROUTER_KEY` (ou setar variável de ambiente).
3. Abrir `Depression_Annotator_jury_LLMs_Laerty copy.ipynb` e executar:
   - Configuração/OpenRouter + modelos,
   - Definição da Signature e teste rápido,
   - Carregamento do `gold_standard_depseverity.csv` e métricas do baseline,
   - Criação de `train_set` / `dev_set`,
   - Compilação DSPy para cada modelo do júri.
