import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==============================================================================
# FUNÇÕES DE LIMPEZA
# ==============================================================================

def clean_weird_number(x):
    """
    Tenta converter strings com múltiplos pontos (ex: '8.223.656...') 
    para float. Estratégia: Considerar o primeiro ponto como decimal.
    """
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s == '': return np.nan
    
    if s.replace('.', '', 1).isdigit():
        return float(s)
    
    parts = s.split('.')
    if len(parts) > 2:
        new_val = parts[0] + '.' + ''.join(parts[1:])
        try:
            return float(new_val)
        except:
            return np.nan
    return pd.to_numeric(x, errors='coerce')

# ==============================================================================
# CARREGAMENTO E SELEÇÃO
# ==============================================================================

file_path = 'ndt_tests_corrigido.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Aviso: Arquivo .csv não encontrado.")
    df = pd.DataFrame(columns=['client', 'download_throughput_bps', 'rtt_download_sec', 
                               'upload_throughput_bps', 'rtt_upload_sec'])

# Aplicar limpeza em todas as colunas numéricas relevantes
numeric_cols = ['download_throughput_bps', 'rtt_download_sec', 
                'upload_throughput_bps', 'rtt_upload_sec', 'packet_loss_percent']

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].apply(clean_weird_number)

if not df.empty:
    # Remover linhas inválidas (Garantir que todas as 4 métricas sejam > 0)
    df_clean = df.dropna(subset=['download_throughput_bps', 'rtt_download_sec',
                                 'upload_throughput_bps', 'rtt_upload_sec'])
    
    df_clean = df_clean[
        (df_clean['download_throughput_bps'] > 0) & 
        (df_clean['rtt_download_sec'] > 0) &
        (df_clean['upload_throughput_bps'] > 0) & 
        (df_clean['rtt_upload_sec'] > 0)
    ].copy()

    print(f"Dados carregados e limpos. Total de registros completos válidos: {len(df_clean)}")
    
    # Seleção Manual dos Clientes
    client_A_name = 'client06'
    client_B_name = 'client10'

    available_clients = df_clean['client'].unique()
    if client_A_name not in available_clients or client_B_name not in available_clients:
        print(f"\nERRO: Clientes {client_A_name} ou {client_B_name} não encontrados.")
        print(f"Clientes disponíveis: {available_clients}")
    else:
        print(f"\n--- Comparando Clientes: {client_A_name} vs {client_B_name} ---")

        data_A = df_clean[df_clean['client'] == client_A_name]
        data_B = df_clean[df_clean['client'] == client_B_name]

        # ==============================================================================
        # 6.1 TESTE DE RAZÃO DE VEROSSIMILHANÇA (LRT) - THROUGHPUT (GAMMA)
        # ==============================================================================
        print("\n" + "="*60)
        print("6.1 Teste de Razão de Verossimilhança - THROUGHPUT (Gama)")
        print("="*60)

        # --- DOWNLOAD ---
        print("\n>>> ANÁLISE DE DOWNLOAD <<<")
        Y_A_down = data_A['download_throughput_bps'].values
        Y_B_down = data_B['download_throughput_bps'].values
        n_A_down, n_B_down = len(Y_A_down), len(Y_B_down)
        Y_pooled_down = np.concatenate([Y_A_down, Y_B_down])
        
        k_hat_down, _, _ = stats.gamma.fit(Y_pooled_down, floc=0)
        mean_A_down = np.mean(Y_A_down)
        mean_B_down = np.mean(Y_B_down)
        mean_pooled_down = np.mean(Y_pooled_down)
        
        term_A_down = n_A_down * np.log(mean_pooled_down / mean_A_down)
        term_B_down = n_B_down * np.log(mean_pooled_down / mean_B_down)
        W_obs_down = 2 * k_hat_down * (term_A_down + term_B_down)
        p_val_down = 1 - stats.chi2.cdf(W_obs_down, df=1)
        
        print(f"k estimado: {k_hat_down:.4f} | Médias: {mean_A_down:.2f} vs {mean_B_down:.2f}")
        print(f"W_obs: {W_obs_down:.4f} | P-valor: {p_val_down:.6e}")
        if p_val_down < 0.05: print(">> REJEITAR H0 (Download diferente)")
        else: print(">> NÃO REJEITAR H0 (Download igual)")

        # --- UPLOAD ---
        print("\n>>> ANÁLISE DE UPLOAD <<<")
        Y_A_up = data_A['upload_throughput_bps'].values
        Y_B_up = data_B['upload_throughput_bps'].values
        n_A_up, n_B_up = len(Y_A_up), len(Y_B_up)
        Y_pooled_up = np.concatenate([Y_A_up, Y_B_up])
        
        k_hat_up, _, _ = stats.gamma.fit(Y_pooled_up, floc=0)
        mean_A_up = np.mean(Y_A_up)
        mean_B_up = np.mean(Y_B_up)
        mean_pooled_up = np.mean(Y_pooled_up)
        
        term_A_up = n_A_up * np.log(mean_pooled_up / mean_A_up)
        term_B_up = n_B_up * np.log(mean_pooled_up / mean_B_up)
        W_obs_up = 2 * k_hat_up * (term_A_up + term_B_up)
        p_val_up = 1 - stats.chi2.cdf(W_obs_up, df=1)
        
        print(f"k estimado: {k_hat_up:.4f} | Médias: {mean_A_up:.2f} vs {mean_B_up:.2f}")
        print(f"W_obs: {W_obs_up:.4f} | P-valor: {p_val_up:.6e}")
        if p_val_up < 0.05: print(">> REJEITAR H0 (Upload diferente)")
        else: print(">> NÃO REJEITAR H0 (Upload igual)")

        # ==============================================================================
        # 6.2 TESTE DE RAZÃO DE VEROSSIMILHANÇA (LRT) - RTT (NORMAL)
        # ==============================================================================
        print("\n" + "="*60)
        print("6.2 Teste de Razão de Verossimilhança - RTT (Normal)")
        print("="*60)

        # --- DOWNLOAD RTT ---
        print("\n>>> ANÁLISE DE RTT (DOWNLOAD) <<<")
        R_A_down = data_A['rtt_download_sec'].values
        R_B_down = data_B['rtt_download_sec'].values
        n_A_rd, n_B_rd = len(R_A_down), len(R_B_down)
        
        var_pooled_rd = np.var(np.concatenate([R_A_down, R_B_down]), ddof=0)
        mean_A_rd = np.mean(R_A_down)
        mean_B_rd = np.mean(R_B_down)
        
        factor_n_rd = (n_A_rd * n_B_rd) / (n_A_rd + n_B_rd)
        diff_sq_rd = (mean_A_rd - mean_B_rd) ** 2
        W_obs_rd = (1 / var_pooled_rd) * factor_n_rd * diff_sq_rd
        p_val_rd = 1 - stats.chi2.cdf(W_obs_rd, df=1)
        
        print(f"Var (Pooled): {var_pooled_rd:.6f} | Médias: {mean_A_rd:.4f} vs {mean_B_rd:.4f}")
        print(f"W_obs: {W_obs_rd:.4f} | P-valor: {p_val_rd:.6e}")
        if p_val_rd < 0.05: print(">> REJEITAR H0 (RTT Down diferente)")
        else: print(">> NÃO REJEITAR H0")

        # --- UPLOAD RTT ---
        print("\n>>> ANÁLISE DE RTT (UPLOAD) <<<")
        R_A_up = data_A['rtt_upload_sec'].values
        R_B_up = data_B['rtt_upload_sec'].values
        n_A_ru, n_B_ru = len(R_A_up), len(R_B_up)
        
        var_pooled_ru = np.var(np.concatenate([R_A_up, R_B_up]), ddof=0)
        mean_A_ru = np.mean(R_A_up)
        mean_B_ru = np.mean(R_B_up)
        
        factor_n_ru = (n_A_ru * n_B_ru) / (n_A_ru + n_B_ru)
        diff_sq_ru = (mean_A_ru - mean_B_ru) ** 2
        W_obs_ru = (1 / var_pooled_ru) * factor_n_ru * diff_sq_ru
        p_val_ru = 1 - stats.chi2.cdf(W_obs_ru, df=1)
        
        print(f"Var (Pooled): {var_pooled_ru:.6f} | Médias: {mean_A_ru:.4f} vs {mean_B_ru:.4f}")
        print(f"W_obs: {W_obs_ru:.4f} | P-valor: {p_val_ru:.6e}")
        if p_val_ru < 0.05: print(">> REJEITAR H0 (RTT Up diferente)")
        else: print(">> NÃO REJEITAR H0")


        # ==============================================================================
        # GRÁFICOS (4 SUBPLOTS)
        # ==============================================================================
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Helper para plotar Gamma
        def plot_gamma(ax, data1, data2, name1, name2, title, xlabel):
            pooled = np.concatenate([data1, data2])
            k, _, _ = stats.gamma.fit(pooled, floc=0)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            
            # Histograma
            ax.hist(data1, bins=30, density=True, alpha=0.5, color='blue', label=f'{name1} (Hist)')
            ax.hist(data2, bins=30, density=True, alpha=0.5, color='orange', label=f'{name2} (Hist)')
            
            # Linhas Teóricas
            x = np.linspace(0, max(pooled.max(), 1), 300)
            ax.plot(x, stats.gamma.pdf(x, a=k, scale=mean1/k), 'b-', lw=2, label=f'Fit {name1}')
            ax.plot(x, stats.gamma.pdf(x, a=k, scale=mean2/k), 'r-', lw=2, label=f'Fit {name2}')
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)

        # Helper para plotar Normal
        def plot_normal(ax, data1, data2, name1, name2, title, xlabel):
            pooled = np.concatenate([data1, data2])
            var = np.var(pooled, ddof=0)
            std = np.sqrt(var)
            mean1, mean2 = np.mean(data1), np.mean(data2)
            
            # Histograma
            ax.hist(data1, bins=30, density=True, alpha=0.5, color='green', label=f'{name1} (Hist)')
            ax.hist(data2, bins=30, density=True, alpha=0.5, color='purple', label=f'{name2} (Hist)')
            
            # Linhas Teóricas
            min_v, max_v = min(data1.min(), data2.min()), max(data1.max(), data2.max())
            x = np.linspace(min_v, max_v, 300)
            ax.plot(x, stats.norm.pdf(x, loc=mean1, scale=std), 'g-', lw=2, label=f'Fit {name1}')
            ax.plot(x, stats.norm.pdf(x, loc=mean2, scale=std), 'm-', lw=2, label=f'Fit {name2}')
            
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)

        # 1. Download Throughput (Gamma) - Superior Esquerdo
        plot_gamma(axes[0, 0], Y_A_down, Y_B_down, client_A_name, client_B_name, 
                   f"Download Speed (Gamma)\np={p_val_down:.1e}", "bps")

        # 2. Upload Throughput (Gamma) - Superior Direito
        plot_gamma(axes[0, 1], Y_A_up, Y_B_up, client_A_name, client_B_name, 
                   f"Upload Speed (Gamma)\np={p_val_up:.1e}", "bps")

        # 3. Download RTT (Normal) - Inferior Esquerdo
        plot_normal(axes[1, 0], R_A_down, R_B_down, client_A_name, client_B_name, 
                    f"Download RTT (Normal)\np={p_val_rd:.1e}", "segundos")

        # 4. Upload RTT (Normal) - Inferior Direito
        plot_normal(axes[1, 1], R_A_up, R_B_up, client_A_name, client_B_name, 
                    f"Upload RTT (Normal)\np={p_val_ru:.1e}", "segundos")

        plt.tight_layout()
        plt.show()

else:
    print("DataFrame vazio. Verifique o arquivo CSV.")