import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def clean_weird_number(x):
    """
    Tenta converter strings com múltiplos pontos (ex: '8.223.656...') 
    para float. Estratégia: Considerar o primeiro ponto como decimal 
    e ignorar o resto, ou tratar como separador de milhar.
    
    Dada a análise do arquivo:
    - '8.223.656...' -> Parece ser 8.22 Mbps ou 8 Gbp. 
      Vamos assumir a estrutura "Primeiro ponto é decimal" para normalizar.
      Isso transforma '90.273...' em 90.273.
    """
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == '': return np.nan
    
    # Se já for um número limpo (ex: '12')
    if s.replace('.', '', 1).isdigit():
        return float(s)
    
    # Caso complexo: '8.223.656.799...'
    # Estratégia: Pegar a parte antes do segundo ponto.
    parts = s.split('.')
    if len(parts) > 2:
        # Reconstrói como "Inteiro.Decimal" usando as duas primeiras partes
        # Ex: 8.223.656 -> 8.223656
        new_val = parts[0] + '.' + ''.join(parts[1:])
        try:
            return float(new_val)
        except:
            return np.nan
    return pd.to_numeric(x, errors='coerce')

# Carregar dataset
file_path = 'ndt_tests_corrigido.csv'
df = pd.read_csv(file_path)

# Aplicar limpeza
numeric_cols = ['download_throughput_bps', 'rtt_download_sec', 
                'upload_throughput_bps', 'rtt_upload_sec', 'packet_loss_percent']

for col in numeric_cols:
    df[col] = df[col].apply(clean_weird_number)

# Remover linhas inválidas (Zeros ou NaNs que quebram logaritmos/divisões)
df_clean = df.dropna(subset=numeric_cols)
df_clean = df_clean[
    (df_clean['download_throughput_bps'] > 0) & 
    (df_clean['rtt_download_sec'] > 0)
].copy()

print(f"Dados carregados e limpos. Total de registros válidos: {len(df_clean)}")
print(df_clean[['download_throughput_bps', 'rtt_download_sec']].head())


# Selecionar os 2 clientes com mais observações para garantir robustez estatística
top_clients = df_clean['client'].value_counts().head(2).index.tolist()

if len(top_clients) < 2:
    raise ValueError("Não há clientes suficientes (mínimo 2) após a limpeza.")

client_A_name = top_clients[0]
client_B_name = top_clients[1]

print(f"\n--- Comparando Clientes: {client_A_name} vs {client_B_name} ---")

data_A = df_clean[df_clean['client'] == client_A_name]
data_B = df_clean[df_clean['client'] == client_B_name]


print("\n" + "="*60)
print("6.1 Teste de Razão de Verossimilhança (LRT) - Throughput (Gama)")
print("="*60)

# Dados
Y_A = data_A['download_throughput_bps'].values
Y_B = data_B['download_throughput_bps'].values
n_A = len(Y_A)
n_B = len(Y_B)

# Estimação de k (shape) conjunto (H0: mesma distribuição Gamma exceto taxa?)
# Para o teste, assumimos k conhecido/fixo. Vamos estimar k usando todos os dados (Pooled).
Y_pooled = np.concatenate([Y_A, Y_B])
# Ajuste Gamma: retorna (alpha/k, loc, scale=1/beta)
k_hat, loc_hat, scale_hat = stats.gamma.fit(Y_pooled, floc=0)

print(f"Parâmetro k (shape) estimado (Pooled): {k_hat:.4f}")

# Médias
mean_A = np.mean(Y_A)
mean_B = np.mean(Y_B)
mean_pooled = np.mean(Y_pooled)

print(f"Média {client_A_name}: {mean_A:.4f}")
print(f"Média {client_B_name}: {mean_B:.4f}")

# Estatística do Teste (W)
# W = 2k [ nA * log(Y_bar / Y_bar_A) + nB * log(Y_bar / Y_bar_B) ]
# Nota: log(Y_pooled/mean_A) é negativo se Y_pooled < mean_A. 
# A estatística de desvio (deviance) deve ser positiva.
term_A = n_A * np.log(mean_pooled / mean_A)
term_B = n_B * np.log(mean_pooled / mean_B)
W_obs_gamma = 2 * k_hat * (term_A + term_B)

# P-valor
# Sob H0, W ~ Chi-quadrado com 1 grau de liberdade
p_value_gamma = 1 - stats.chi2.cdf(W_obs_gamma, df=1)
chi2_crit = stats.chi2.ppf(0.95, df=1)

print(f"\nEstatística W_obs: {W_obs_gamma:.4f}")
print(f"Valor Crítico (95%): {chi2_crit:.4f}")
print(f"P-valor: {p_value_gamma:.6e}")

if W_obs_gamma > chi2_crit:
    print(">> REJEITAR H0: As taxas de throughput são significativamente diferentes.")
else:
    print(">> NÃO REJEITAR H0: Não há diferença significativa nas taxas.")


print("\n" + "="*60)
print("6.2 Teste de Razão de Verossimilhança (LRT) - RTT (Normal)")
print("="*60)

# Dados
R_A = data_A['rtt_download_sec'].values
R_B = data_B['rtt_download_sec'].values
n_A_rtt = len(R_A)
n_B_rtt = len(R_B)

# Estimação da variância (sigma^2) conjunta (Pooled)
# O teste assume variância conhecida. Usamos a variância amostral conjunta como estimativa.
var_pooled = np.var(np.concatenate([R_A, R_B]), ddof=0) 
sigma2_hat = var_pooled

# Médias
mean_A_rtt = np.mean(R_A)
mean_B_rtt = np.mean(R_B)

print(f"Média RTT {client_A_name}: {mean_A_rtt:.4f}")
print(f"Média RTT {client_B_name}: {mean_B_rtt:.4f}")
print(f"Variância (sigma^2): {sigma2_hat:.4f}")

# Estatística do Teste (W)
# W = (1/sigma^2) * (nA*nB / nA+nB) * (MeanA - MeanB)^2
factor_n = (n_A_rtt * n_B_rtt) / (n_A_rtt + n_B_rtt)
diff_sq = (mean_A_rtt - mean_B_rtt) ** 2
W_obs_normal = (1 / sigma2_hat) * factor_n * diff_sq

# P-valor
p_value_normal = 1 - stats.chi2.cdf(W_obs_normal, df=1)

print(f"\nEstatística W_obs: {W_obs_normal:.4f}")
print(f"Valor Crítico (95%): {chi2_crit:.4f}")
print(f"P-valor: {p_value_normal:.6e}")

if W_obs_normal > chi2_crit:
    print(">> REJEITAR H0: As médias de RTT são significativamente diferentes.")
else:
    print(">> NÃO REJEITAR H0: Não há diferença significativa nas médias de RTT.")

print("\n" + "="*60)