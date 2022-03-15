#-----------------------------------------------------------------------------------------------------------------------------------
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as snsp
import pandas_datareader.data as web
from datetime import datetime, date, timedelta
import yfinance as yf
from fitter import Fitter, get_common_distributions, get_distributions
import dash 
from dash.dependencies import Input, Output
import plotly.express as px
import copy
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
#Getting Data
acoes = open("tickers.txt", "r", encoding="utf8").read().splitlines()
dados_f = pd.DataFrame()
end = datetime.today()
start = end - timedelta(450)
end = end.strftime("%Y-%m-%d")
start = start.strftime("%Y-%m-%d")
dados= yf.download(tickers = acoes, start = start, end = end)['Adj Close']
dados = dados.tail(260)


#Liquidity and NaN filter
dados_f = dados.copy()
dados_f.dropna(axis=0, how="all", inplace=True)

acoes_f = list(dados_f.columns)

liquidez_desejada = 0.95 
for a in acoes_f:
    if dados_f.loc[:,a].describe().loc['count']<liquidez_desejada*len(dados_f.index):
        dados_f.drop(columns = [a], inplace = True)

acoes_f = list(dados_f.columns)
acoes_f_sem_bvsp = copy.deepcopy(acoes_f)
acoes_f_sem_bvsp.remove("^BVSP")
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
#Calculating Return Metrics
retornos_diarios = dados_f/dados_f.shift(1)
retornos_diarios_med = retornos_diarios.mean()
retornos_diarios_med = retornos_diarios_med.rename('Retornos diários médios')
ret_min = pd.Series(dtype = float)
ret_max = pd.Series(dtype = float)
ret_min.reindex_like(retornos_diarios_med)
ret_max.reindex_like(retornos_diarios_med)
for a in acoes_f:
    ret_min[a] = retornos_diarios[a].min()-1
    ret_max[a] = retornos_diarios[a].max()-1
ret_min = ret_min.rename("Retorno Mínimo")
ret_max = ret_max.rename("Retorno Máximo")
ret_acum=dados_f.iloc[-1]/dados_f.iloc[0]
ret_acum=ret_acum.rename("Retorno líquido acumulado")
retorno = pd.DataFrame()
retorno = retorno.append(retornos_diarios_med)
retorno = retorno.append(ret_min)
retorno = retorno.append(ret_max)
retorno = retorno.append(ret_acum)
del retorno['^BVSP']
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
#Calculating Risk Metrics

#Average Deviation
desvio_padrao = retornos_diarios.std()
desvio_padrao = desvio_padrao.rename('Desvio Padrão')

#Semivariance
def f_semivariancia (retornos_diarios, media):
    tamanho=len(retornos_diarios)
    somatorio = 0
    #print('Média: ' + str(media))                                   // para debug
    #print('Retorno no dia 1: ' + str(retornos_diarios.iloc[1]))     // para debug
    t = 0
    while t in range(tamanho):
        dif = (retornos_diarios.iloc[t] - media)
        #print('Dif = ' + str(retornos_diarios.iloc[t] - media))    // para debug
        if dif < 0:
            somatorio += dif ** 2
        t += 1
    med = somatorio / tamanho
    semivariancia = med ** (0.5)
    return (semivariancia)
semivariancia = []
for a in acoes_f:
    semivariancia.append(f_semivariancia(retornos_diarios[a],retornos_diarios_med.loc[a]))
semivariancia
semivariancia = pd.Series(semivariancia)
semivariancia.index = acoes_f
semivariancia = semivariancia.rename('Semivariância')

#Market Beta

cov = retornos_diarios.cov()

beta = []
for a in acoes_f:
    beta.append(cov.loc[a, '^BVSP']/retornos_diarios['^BVSP'].var())
beta = pd.Series(beta)
beta.index = acoes_f
beta = beta.rename('Beta de mercado')
beta

#Downside Risk
def f_downside (retornos_diarios_acao, retornos_diarios_bovespa):
    tamanho=len(retornos_diarios)
    t = 0
    dif = (retornos_diarios_acao - retornos_diarios_bovespa)
    while t in range(tamanho):
        if dif.iloc[t] > 0:
            dif.iloc[t] = 0
        t += 1
    dif = dif ** 2
    media = dif.mean()
    Downside_Risk = media ** (0.5)
    return (Downside_Risk)
Downside_Risk = []
for a in acoes_f:
    Downside_Risk.append(f_downside(retornos_diarios[a],retornos_diarios['^BVSP']))
Downside_Risk = pd.Series(Downside_Risk)
Downside_Risk.index = acoes_f
Downside_Risk = Downside_Risk.rename('Downside Risk')

#Tracking Error Volatility
TE = retornos_diarios.sub(retornos_diarios['^BVSP'], axis=0)
TEV = TE.std()
TEV.rename("Tracking Error Volatility", inplace = True)

#Drawdown
RBA = pd.DataFrame()
serie_unit = []
tamanho_dias = len(retornos_diarios)
a = 0
while a in range(len(acoes_f)):
    serie_unit.append(1)
    a += 1
serie_unit = pd.Series(serie_unit)
serie_unit.index = acoes_f
RBA = RBA.reindex_like(retornos_diarios, method = 'backfill')
RBA.iloc[0] = serie_unit
for a in acoes_f:
    t=1
    while t in range(tamanho_dias):
        RBA[a].iloc[t]=RBA[a].iloc[t-1]*retornos_diarios[a].iloc[t]
        t+=1
drawdown = pd.DataFrame()
drawdown = drawdown.reindex_like(retornos_diarios, method = 'backfill')
#min(RBA['WEGE3.SA'].tail(tamanho_dias-200))
for a in acoes_f:
    t=0
    while t in range(tamanho_dias):
        drawdown[a].iloc[t]=(RBA[a].iloc[t]-min(RBA[a].tail(tamanho_dias-t)))/RBA[a].iloc[t]
        t+=1
drawdown_max = []
for a in acoes_f:
    drawdown_max.append(max(drawdown[a]))
drawdown_max = pd.Series(drawdown_max)
drawdown_max.index = acoes_f
drawdown_max = drawdown_max.rename('Drawdown')

#Risk Dataframe
risco = pd.DataFrame()
risco = risco.append(desvio_padrao)
risco = risco.append(semivariancia)
risco = risco.append(drawdown_max)
risco = risco.append(Downside_Risk)
risco = risco.append(TEV)
risco = risco.append(beta)
del risco['^BVSP']

#Normalized Risk Dataframe
media_dp = risco.loc['Desvio Padrão'].mean()
dp_dp = risco.loc['Desvio Padrão'].std()
media_sv = risco.loc['Semivariância'].mean()
dp_sv = risco.loc['Semivariância'].std()
media_drd = risco.loc['Drawdown'].mean()
dp_drd = risco.loc['Drawdown'].std()
media_dsr = risco.loc['Downside Risk'].mean()
dp_dsr = risco.loc['Downside Risk'].std()
media_tev = risco.loc['Tracking Error Volatility'].mean()
dp_tev = risco.loc['Tracking Error Volatility'].std()
media_beta = risco.loc['Beta de mercado'].mean()
dp_beta = risco.loc['Beta de mercado'].std()

risco_norm = pd.DataFrame()
dp_norm = (risco.loc['Desvio Padrão'] - media_dp) / dp_dp
dp_norm = dp_norm.rename('Desvio Padrão Normalizado')
risco_norm = risco_norm.append(dp_norm)
sv_norm = (risco.loc['Semivariância'] - media_sv) / dp_sv
sv_norm = sv_norm.rename('Semivariância Normalizada')
risco_norm = risco_norm.append(sv_norm)
drd_norm = (risco.loc['Drawdown'] - media_drd) / dp_drd
drd_norm = drd_norm.rename('Drawdown Normalizado')
risco_norm = risco_norm.append(drd_norm)
dsr_norm = (risco.loc['Downside Risk'] - media_dsr) / dp_dsr
dsr_norm = dsr_norm.rename('Downside Risk Normalizado')
risco_norm = risco_norm.append(dsr_norm)
tev_norm = (risco.loc['Tracking Error Volatility'] - media_tev) / dp_tev
tev_norm = tev_norm.rename('Tracking Error Volatility Normalizado')
risco_norm = risco_norm.append(tev_norm)
beta_norm = (risco.loc['Beta de mercado'] - media_beta) / dp_beta
beta_norm = beta_norm.rename('Beta de mercado Normalizado')
risco_norm = risco_norm.append(beta_norm)

#Risk Score
nota_risco = []
t=0
for a in (acoes_f_sem_bvsp):
    nota_risco.append(risco_norm[a].mean())
nota_risco = pd.Series(nota_risco)
nota_risco.index = acoes_f_sem_bvsp
nota_risco = nota_risco.sort_values(ascending = False)

#Risk Ranking
ranking_risco = []
t=0
while t in range(len(nota_risco)):
    ranking_risco.append(t+1)
    t+=1
ranking_risco = pd.Series(ranking_risco)
ranking_risco.index = nota_risco.index
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
#Calculating Performance Metrics

##Várias métricas para Risk Free
ret_poup = 0.0072217938876129/100

CDI = 4.15/100
ret_nu = (1+CDI)**(1/360)-1

##Definir qual é o Risk Free
risk_free = ret_nu

#Sharpe Index
sharpe = []
for a in acoes_f:
    sharpe.append((retornos_diarios_med[a]-1-risk_free)/desvio_padrao[a])
sharpe = pd.Series(sharpe)
sharpe.index = acoes_f
sharpe = sharpe.rename('Índice de Sharpe')

#Sortino Index - Semivariance
sortino_sv = []
for a in acoes_f:
    sortino_sv.append((retornos_diarios_med[a]-1-risk_free)/semivariancia[a])
sortino_sv = pd.Series(sortino_sv)
sortino_sv.index = acoes_f
sortino_sv = sortino_sv.rename('Sortino (semivariância)')

#Sortino Index - Downside Risk
sortino_dsr = []
for a in acoes_f:
    sortino_dsr.append((retornos_diarios_med[a]-retornos_diarios_med['^BVSP'])/Downside_Risk[a])
sortino_dsr = pd.Series(sortino_dsr)
sortino_dsr.index = acoes_f
sortino_dsr = sortino_dsr.rename('Sortino (Downside Risk)')

#Treynor Index
Treynor = []
for a in acoes_f:
    Treynor.append((retornos_diarios_med[a]-1-risk_free)/beta[a])
Treynor = pd.Series(Treynor)
Treynor.index = acoes_f
Treynor = Treynor.rename('Índice de Treynor')

#Calmar Index
Calmar = []
for a in acoes_f:
    Calmar.append((retornos_diarios_med[a]-1-risk_free)/drawdown_max[a])
Calmar = pd.Series(Calmar)
Calmar.index = acoes_f
Calmar = Calmar.rename('Índice de Calmar')

#Information Ration
InfRatio = []
for a in acoes_f:
    InfRatio.append((retornos_diarios_med[a]-retornos_diarios_med['^BVSP'])/TEV[a])
InfRatio = pd.Series(InfRatio)
InfRatio.index = acoes_f
InfRatio = InfRatio.rename('Information Ratio')

#M2
M2 = []
for a in acoes_f:
    M2.append(sharpe[a]*desvio_padrao['^BVSP']+risk_free)
M2 = pd.Series(M2)
M2.index = acoes_f
M2 = M2.rename('M2')

#Performance Dataframe
perf = pd.DataFrame()
perf = perf.append(sharpe)
perf = perf.append(sortino_sv)
perf = perf.append(sortino_dsr)
perf = perf.append(Treynor)
perf = perf.append(Calmar)
perf = perf.append(InfRatio)
perf = perf.append(M2)
del perf['^BVSP']

#Normalized Performance Dataframe
media_sharpe = perf.loc['Índice de Sharpe'].mean()
dp_sharpe = perf.loc['Índice de Sharpe'].std()
media_sort_sv = perf.loc['Sortino (semivariância)'].mean()
dp_sort_sv = perf.loc['Sortino (semivariância)'].std()
media_sort_dsr = perf.loc['Sortino (Downside Risk)'].mean()
dp_sort_dsr = perf.loc['Sortino (Downside Risk)'].std()
media_treynor = perf.loc['Índice de Treynor'].mean()
dp_treynor = perf.loc['Índice de Treynor'].std()
media_calmar = perf.loc['Índice de Calmar'].mean()
dp_calmar = perf.loc['Índice de Calmar'].std()
media_ir = perf.loc['Information Ratio'].mean()
dp_ir = perf.loc['Information Ratio'].std()
media_m2 = perf.loc['M2'].mean()
dp_m2 = perf.loc['M2'].std()

perf_norm = pd.DataFrame()
##sharpe
sharpe_norm = (perf.loc['Índice de Sharpe'] - media_sharpe) / dp_sharpe
sharpe_norm = sharpe_norm.rename('Sharpe Normalizado')
perf_norm = perf_norm.append(sharpe_norm)
##sortino sv
ssv_norm = (perf.loc['Sortino (semivariância)'] - media_sort_sv) / dp_sort_sv
ssv_norm = ssv_norm.rename('Sortino (semivariância) Normalizado')
perf_norm = perf_norm.append(ssv_norm)
##sortino dsr
sdsr_norm = (perf.loc['Sortino (Downside Risk)'] - media_sort_dsr) / dp_sort_dsr
sdsr_norm = sdsr_norm.rename('Sortino (Downside Risk) Normalizado')
perf_norm = perf_norm.append(sdsr_norm)
##Treynor
treynor_norm = (perf.loc['Índice de Treynor'] - media_treynor) / dp_treynor
treynor_norm = treynor_norm.rename('Treynor Normalizado')
perf_norm = perf_norm.append(treynor_norm)
##Calmar
calmar_norm = (perf.loc['Índice de Calmar'] - media_calmar) / dp_calmar
calmar_norm = calmar_norm.rename('Calmar Normalizado')
perf_norm = perf_norm.append(calmar_norm)
##IR
ir_norm = (perf.loc['Information Ratio'] - media_ir) / dp_ir
ir_norm = ir_norm.rename('Information Ratio Normalizado')
perf_norm = perf_norm.append(ir_norm)
##M2
m2_norm = (perf.loc['M2'] - media_m2) / dp_m2
m2_norm = sharpe_norm.rename('M2 Normalizado')
perf_norm = perf_norm.append(m2_norm)

#Performance Score
nota_perf = []
t=0
for a in (acoes_f_sem_bvsp):
    nota_perf.append(perf_norm[a].mean())
nota_perf = pd.Series(nota_perf)
nota_perf.index = acoes_f_sem_bvsp
nota_perf = nota_perf.sort_values(ascending = False)

#Performance Ranking
ranking_perf = []
t=0
while t in range(len(nota_perf)):
    ranking_perf.append(t+1)
    t+=1
ranking_perf = pd.Series(ranking_perf)
ranking_perf.index = nota_perf.index
#-----------------------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------------------
#Importing "Dados_Cadastrais.xlsx"