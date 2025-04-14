import matplotlib.pyplot as plt
import numpy
import json
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def ler_nano(ficheiro):
    '''
    Função para ler valores numéricos de um arquivo .nano e separar os valores em duas listas: x_values e y_values.
    Após isto retorna os valores da amplitude (X) e o dicionário com as características do sinal emitido

    @param: str file - O nome do ficheiro texto .nano
    @return: tuple - Tuple formada por: Valores da Amplitude e Dicionario obtido através do JSON
    ''' 

    # Abrir o ficheiro e ler a primeira linha (JSON)
    file=open(ficheiro,"r")
    y=file.readline()
    posicao=y.find("}",1) # Encontrar a posicao da primeira chaveta "}" - Fim do JSON
    posicao=y.find("}",posicao)
    y=y[0:posicao+1] # Apagar da posição seguinte para a frente
    dicionario_detalhes=json.loads(y)   # Converter JSON para dicionário Python
    file.close() # Fechar Ficheiro


    # Abrir o ficheiro e ler as restantes linhas (valores)
    file=open(ficheiro,"r")
    numeros=file.readlines() #Abrir seguintes linhas de código
    file.close() # Fechar Ficheiro
    numeros=numeros[1:]
    x_values=[] # Inicializar lista dos valores de x
    y_values=[] # Inicializar lista dos valores de y

    for linha in numeros:
        valores=linha.strip().split('\t') # Formatar string para um float de modo a traçar gráfico

        if abs(float(valores[0]))<1:
            x_values.append(float(valores[0]))
        y_values.append(float(valores[1]))
   
    # A minha ideia inicial era inverter o gráfico para comparar com a imagem do tubo mas não era prático, pelo que comentei esta seccao toda

    # if dicionario_detalhes["Direction"]=="Left": # Se o sinal for emitido para a esquerda temos de inverter o gráfico
    #     x_values.reverse()

    return x_values,dicionario_detalhes # Valores do 2ºCanal(y) não são necessários


# Obter informações sobre os sinais e criar vetor distancia
sinal1,dicionario1=ler_nano("medida1.nano")
t1=numpy.linspace(0,len(sinal1)*5e-7,len(sinal1))
distancia1=t1*0.5*3200
sinal2,dicionario2=ler_nano("medida2.nano")
t2=numpy.linspace(0,len(sinal2)*5e-7,len(sinal2))
distancia2=t2*0.5*3200
sinal3,dicionario3=ler_nano("medida3.nano")
t3=numpy.linspace(0,len(sinal3)*5e-7,len(sinal3))
distancia3=t3*0.5*3200

sinal1=numpy.array(sinal1)  # Converter para array, para puder manipular os dados mais facilmente
sinal2=numpy.array(sinal2)  # Converter para array, para puder manipular os dados mais facilmente
sinal3=numpy.array(sinal3)  # Converter para array, para puder manipular os dados mais facilmente

# Encontrar os picos
picos1,propriedades=find_peaks(sinal1, height=0.05)  # ajusta o valor de height - Aplicado a olho, desde que apanhasse os picos desejados serviria
picos2,propriedades=find_peaks(sinal2, height=0.02)  # ajusta o valor de height - Aplicado a olho, desde que apanhasse os picos desejados serviria
picos3,propriedades=find_peaks(sinal3, height=0.05)  # ajusta o valor de height - Aplicado a olho, desde que apanhasse os picos desejados serviria

# Visualizar o sinal e os picos
pen1=picos1[-2] # Marcar indices dos picos - Através de experiência constatei que eram o penúltimo ponto
pen2=picos2[-2] # Marcar índice dos picos - Através de experiência constatei que eram o penúltimo ponto
pen3=picos3[-2] # Marcar índice dos picos - Através de experiência constatei que eram o penúltimo ponto

distancia=[0,pen1*5e-7*3200*0.5,pen2*5e-7*3200*0.5,pen3*5e-7*3200*0.5] # distancias relativas aos picos + valor 0
amplitudes=[1,sinal1[pen1]*4,sinal2[pen2]*4,sinal3[pen3]*4] # amplitudes relativas aos picos + valor 1
distancia_reais=[round(float(v),3) for v in distancia] # Converter de np.float para real
amplitudes_reais=[round(float(v),3) for v in amplitudes] # Converter de np.float para real
distancia_reais.sort() # Ordenar distancias
amplitudes_reais.sort() # Ordenar amplitudes
distancia_reais.reverse() # Reverter distâncias - Dita a direção da DAC Curve

#print(amplitudes_reais) - Uso para teste
#print(distancia_reais) - Uso para teste

# Criar função que me permite interpolar valores da DAC
def modelo_exponencial(d,A0,alpha):
    return A0*numpy.exp(-alpha*d) # Fórmula exponencial - distancia(d) e Coeficiente de atenuação(alpha),Amplitude referência (Ao)

# Ajustar a curva usando os dados corretos
parametros, _=curve_fit(modelo_exponencial,distancia_reais,amplitudes_reais)
A0,alpha=parametros

# Criar vetor de distâncias para toda a extensão da onda (usa por exemplo distancia1)
dist_total=numpy.linspace(min(distancia1),30,1000) # Os valores do meu sinal nunca passam dos 30metros
amp_dac=modelo_exponencial(dist_total,A0,alpha)

# Plot dos sinais + curva DAC
plt.figure(figsize=(14, 6))
plt.xlim(left=0,right=distancia1[-1])
plt.plot(distancia1,sinal1,color='blue',label='Sinal 1')
plt.plot(distancia2,sinal2,color='green',label='Sinal 2')
plt.plot(distancia3,sinal3,color='red',label='Sinal 3')

# Marcar os picos usados para ajustar a DAC
plt.scatter(distancia_reais,amplitudes_reais,color='black',s=60,marker='x',label='Picos usados')

# Curva DAC ao longo de toda a distância
plt.plot(dist_total,amp_dac,color='orange',linewidth=2.5,label='Curva DAC')
plt.xlabel("Distância (m)")
plt.ylabel("Amplitude (Normalizada)")
plt.title("Ensaios com Curva DAC (Interpolação)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Guardar curva DAC num ficheiro texto para depois utilizar noutros códigos
with open("curva_dac.txt", "w") as f:
    f.write("Distancia(m)\tAmplitude\n")
    for d, a in zip(dist_total, amp_dac):
        f.write(f"{d:.4f}\t{a:.4f}\n")


