import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
dados = pd.read_csv('./sinasc_2016.csv')

# Correção de tipos de dados
dados['DTNASC'] = pd.to_datetime(dados['DTNASC'])
dados['DTNASCMAE'] = pd.to_datetime(dados['DTNASCMAE'])

# Distribuição das variáveis
print("Distribuição do código de município de nascimento:")
print(dados["CODMUNNASC"].describe())

print("\nDistribuição do código de município de residência:")
print(dados["CODMUNRES"].describe())

# Agrupando por município e contando os nascimentos
municipios_por_uf = dados["CODMUNRES"].value_counts()

# Mostrando os resultados
print("\nQuantidade de nascimentos por município de residência:")
print(municipios_por_uf)

# Calculando a idade da mãe
dados['IDADEMAE'] = (dados['DTNASC'] - dados['DTNASCMAE']).dt.days // 365

# Idade mínima das mães
idade_min_mae = dados["IDADEMAE"].min()

# Idade máxima das mães
idade_max_mae = dados["IDADEMAE"].max()

# Mostrando os resultados
print("\nTrês menores idades das mães:", dados["IDADEMAE"].sort_values().head(3).tolist())
print("\nTrês maiores idades das mães:", dados["IDADEMAE"].sort_values().tail(3).tolist())

# Filtrando dados da Bahia (código 29)
bahia = dados[dados["CODMUNRES"].astype(str).str.startswith('29')]

# Distribuição da idade das mães na Bahia
print("\nDistribuição de frequência da idade das mães na Bahia:")
print(bahia["IDADEMAE"].describe())

# Distribuição da variável 'PESO'
print("\nDistribuição da variável 'PESO':")
print(dados["PESO"].describe())

# Filtrando bebês com peso inferior a 2000 gramas
abaixo_2000 = dados[dados["PESO"] < 2000]

# Quantidade de bebês
qnt_abaixo_2000 = abaixo_2000.shape[0]

# Porcentagem em relação ao total
porcentagem_abaixo_2000 = (qnt_abaixo_2000 / dados.shape[0]) * 100

# Mostrando os resultados
print("\nQuantidade de bebês nascidos pesando menos de 2000 gramas:", qnt_abaixo_2000)
print("\nPorcentagem:", porcentagem_abaixo_2000, "%")

# Agrupando por UF e contando os estabelecimentos de saúde
estabelecimentos_por_uf = dados["CODESTAB"].value_counts()

# Mostrando os resultados
print("\nQuantidade de estabelecimentos de saúde por UF:")
print(estabelecimentos_por_uf)

# Filtrando nascimentos em hospital na Bahia
hospital_bahia = dados[(dados["CODMUNRES"].astype(str).str.startswith('29')) & (dados["LOCNASC"] == 1)]

# Quantidade de nascimentos
qnt_hospital_bahia = hospital_bahia.shape[0]

# Mostrando os resultados
print("\nQuantidade de crianças nascidas em hospital na Bahia:", qnt_hospital_bahia)

# Filtrando nascimentos em casa na Bahia
casa_bahia = dados[(dados["CODMUNRES"].astype(str).str.startswith('29')) & (dados["LOCNASC"] == 2)]

# Quantidade de nascimentos
qnt_casa_bahia = casa_bahia.shape[0]

# Mostrando os resultados
print("\nQuantidade de crianças nascidas em casa na Bahia:", qnt_casa_bahia)

# Filtrar para mães solteiras, negras e pardas
maes_especificas = dados[(dados['ESTCIVMAE'] == 1) & 
                         (dados['RACACORMAE'].isin([2, 3]))]

# Criar um boxplot para visualizar a distribuição
plt.figure(figsize=(12, 6))
sns.boxplot(x='CODMUNRES', y='ESCMAE2010', hue='RACACORMAE', data=maes_especificas)
plt.title('Distribuição da escolaridade de mães solteiras, negras e pardas por município')
plt.xticks(rotation=45)
plt.show()

# Filtrar para mulheres casadas com parto vaginal antes de 37 semanas
casadas_vaginal_antes_37 = dados[(dados['ESTCIVMAE'] == 2) & 
                                  (dados['PARTO'] == 1) & 
                                  (dados['SEMAGESTAC'] < 37)]

# Calcular a porcentagem
porcentagem_vaginal = len(casadas_vaginal_antes_37) / len(dados[dados['ESTCIVMAE'] == 2]) * 100
print(f"Porcentagem de mulheres casadas com parto vaginal antes de 37 semanas: {porcentagem_vaginal:.2f}%")

# Filtrar para mulheres casadas com parto cesárea antes de 37 semanas
casadas_cesarea_antes_37 = dados[(dados['ESTCIVMAE'] == 2) & 
                                  (dados['PARTO'] == 2) & 
                                  (dados['SEMAGESTAC'] < 37)]

# Calcular a porcentagem
porcentagem_cesarea = len(casadas_cesarea_antes_37) / len(dados[dados['ESTCIVMAE'] == 2]) * 100
print(f"Porcentagem de mulheres casadas com parto cesárea antes de 37 semanas: {porcentagem_cesarea:.2f}%")

# Extra: Distribuição do Número de Nascimentos por Mês
# Extraindo o mês de nascimento
dados['mes_nascimento'] = dados['DTNASC'].dt.month

# Contando o número de nascimentos por mês
nascimentos_por_mes = dados['mes_nascimento'].value_counts().sort_index()

# Criando gráficos com a distribuição de nascimentos por mês
plt.figure(figsize=(10, 6))
sns.barplot(x=nascimentos_por_mes.index, y=nascimentos_por_mes.values, palette='viridis')
plt.title('Distribuição do Número de Nascimentos por Mês')
plt.xlabel('Mês')
plt.ylabel('Número de Nascimentos')
plt.show()

# Extra 2: Correlação entre Idade da Mãe e Peso do Bebê
# Filtrando para evitar valores faltantes
dados_corr = dados[['IDADEMAE', 'PESO']].dropna()

# Calculando a correlação
correlacao = dados_corr.corr()

# Mostrando os resultados
print(f"Correlação entre a idade da mãe e o peso do bebê:\n{correlacao}")

# Criando gráficos com a correlação
plt.figure(figsize=(10, 6))
sns.scatterplot(x='IDADEMAE', y='PESO', data=dados_corr)
plt.title('Correlação entre a Idade da Mãe e o Peso do Bebê')
plt.xlabel('Idade da Mãe')
plt.ylabel('Peso do Bebê')
plt.show()
