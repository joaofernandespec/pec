import json
import matplotlib.pyplot as plt
import numpy
import scipy.signal as signal


def ler_nano(ficheiro):
    '''
    Função para ler valores numéricos de um arquivo .nano e separar os valores em duas listas: x_values e y_values.
    Após isto retorna os valores da amplitude (X) e o dicionário com as características do sinal emitido

    @param: str file - O nome do ficheiro texto .nano
    @return: tuple - Tuple formada por: Valores da Amplitude e Dicionario obtido através do JSON
    ''' 

    # Abrir o ficheiro e ler a primeira linha (JSON)
    file=open("Teste2.nano","r")
    y=file.readline()
    posicao=y.find("}",1)   # Encontrar a posicao da primeira chaveta "}" - Fim do JSON
    posicao=y.find("}",posicao)
    y=y[0:posicao+1]        # Apagar da posição seguinte para a frente
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

    if dicionario_detalhes["Direction"]=="Left": # Se o sinal for emitido para a esquerda temos de inverter o gráfico
        x_values.reverse()

    return x_values,dicionario_detalhes # Valores do 2ºCanal(y) não são necessários


def grafico(amplitude,dados_fea):
    """
    Cria um gráfico onde X=Distância Percorrida pelo som e Y=Amplitude de Sinal. Traça, também, os locais das anomalias
    
    @param amplitude: Lista com os valores da amplitude (V)
    @param frequencia: Frequência do sinal (Hz)
    @return: tuple - Gráfico MATplotLib
    """
    # Cálculo do período e distância
    # Criar uma lista de numeros inteiros [1,2,3,4...]
    x=[]
    for i in range(1,len(amplitude)+1):
        x.append(i)
    
    x=numpy.array(x,dtype=int)

    # Periodo de leitura = 1 / frequencia # Podemos assumir que é 5e-7s
    periodo=5e-7
    velocidade=3200  # Velocidade do som (m/s) no material
    distancia=velocidade*periodo*x*0.5 # Converter índices para distâncias
    fig,ax=plt.subplots(figsize=(12,10))

    # Plot
    plt.plot(distancia,amplitude)
    plt.xlabel("Distância (m)", fontsize=12)
    plt.ylabel("Amplitude (V)", fontsize=12)
    plt.title("Sinal Sonoro")
    #plt.xlim(left=0)
    
    def tracar_defeitos(dados):
        '''
        Função para indicar defeitos e respetivos locais

        @param: str dados_fea - tuple constituida por par defeito-local
        @return: Plot gráfico dos locais das anomalias 
        '''
        
        defeitos=[] # Inicializar lista de defeitos
        locais=[] # Inicializar lista de locais referentes aos defeitos
        
        # Associar valores às respetivas listas
        for i in range(0,len(dados_fea)):
            defeitos.append(dados_fea[i][0])
            locais.append(int(dados_fea[i][1]))

        for i in range(0,len(locais)):
            ax.axvline(x=locais[i], color="r", linewidth="1.5")
    
    tracar_defeitos(dados_fea)
    return plt.show()


def ler_fea(ficheiro):
    '''
    Função para indicar onde se encontram as anomalias do ensaio teste

    @param: str file - O nome do ficheiro texto .fea
    @return: tuple - Tuple formada por duas listas: defeitos e local_defeitos
    '''
    # Criar dicionário com o objetivo de indicar o defeito
    dicionario_anomalias={
        "W": "Weld",
        "PS": "Suport",
        "F": "Flange",
        "T": "Tee_Branch",
        "EW": "Elbow",
        "B": "Brace",
        "D": "Defect",
        "I": "Indication",
        "V": "Valves",
        "E": "Sensor"

    }

    # Abrir e ler o ficheiro .fea para encontrar anomalias
    file=open(ficheiro,"r")
    y=file.read()

    posicao=y.find("\"")   # Encontrar a posicao da primeira "
    posicao=y.find("\"",posicao+1) # Encontrar a posicao da segunda "
    y=y[posicao+2:] # Eliminar Header
    y=y.split("\n") # Separar por linhas
    posicao=y.index("") # Indicar posicao da linha vazai
    y=y[0:posicao] # Apagar tudo exceto os dados - Não vai ser necessário
   
    file.close() # Fechar ficheiro .fea

    # Abrir e ler o ficheiro .fea para encontrar posicao do sensor
    file=open(ficheiro,"r")
    z=file.read()

    posicao_sensor=z.find("[Sensors]") # Encontrar a posição de "[Sensores]"
    posicao_sensor=z.find("_",posicao_sensor) # Encontrar _ de [Sensors] para a frente
    z=z[posicao_sensor-1:]
    z=z.split("\n") # Separar por linhas
    posicao_=z.index("") # Indicar posicao da linha vazia
    z=z[0:posicao_] # Apagar tudo exceto os dados - Não vai ser necessário

    file.close() # Fechar ficheiro .fea



    result=[] # Incializar lista que irá ser retornada
    sensores=[] # Inicializar lista que tens posicao dos sensores
    
    # Manipular strings
    for item in y:
        name, values=item.split(" = ")
        first_value=values.split("\t")[0].replace('"', '').replace(',', '.')

        if "_" in name: # Esta condição, em principio, vai se verificar sempre - Medida contra possíveis erros
            posicao_=name.find("_") # Encontrar _ que separa tipo de defeito e nº daquele tipo de defeito
            
            # Trabalhar string
            key=name[0:posicao_]
            tipo_anomalia=dicionario_anomalias[key] # Traduz key-valor
            name=tipo_anomalia
            result.append((name, float(first_value))) # Anexar ao result que irá ser retornado
    
    for item in z:
        name, values=item.split(" = ")
        first_value=values.split("\t")[0].replace('"', '').replace(',', '.')

        if "_" in name: # Esta condição, em principio, vai se verificar sempre - Medida contra possíveis erros
            posicao_=name.find("_") # Encontrar _ que separa tipo de defeito e nº daquele tipo de defeito
            
            # Trabalhar string
            key=name[0:posicao_]
            tipo_anomalia=dicionario_anomalias[key] # Traduz key-valor
            name=tipo_anomalia
            sensores.append((name, float(first_value))) # Anexar ao result que irá ser retornado


    return result,sensores


def grafico_fft_anomalias(sinal,anomalias):
    '''
    Função para traçar espectro do domínio temporal e das frequências

    @param: list sinal - lista com valores de amplitude lidos por MOT
    @param: lista sinal - lista com locais das anomalias

    @return: tuple - Gráficos com espectros do domínio temporal e das frequências
    '''    
    # Computar sinal
    N=len(sinal)  # Número de pontos
    T=5e-7  # Período de amostragem
    fs=1/T # Frequência de amostragem
    t=numpy.linspace(0, N*T, N)  # Criar vetor de tempo
    distancia=t*3200*0.5

    # Computar FFT
    fourier=numpy.fft.fft(sinal,N) # FFT
    PSD = (numpy.abs(fourier) ** 2) / (N*T) # Densidade Espectral de Potência
    frequencias=numpy.fft.fftfreq(N,T) # Vetor frequencias
    L=numpy.arange(0,numpy.floor(N/2),dtype='int') # So queremos metade do espectro
    magnitude = numpy.abs(fourier) / max(numpy.abs(fourier)) # Normalizar

    # Plotar grafico
    fig,[ax1,ax2]=plt.subplots(nrows=2,ncols=1)
    plt.sca(ax1)
    #plt.plot(t,sinal)
    plt.plot(distancia,sinal)

    plt.sca(ax2)
    #plt.plot(frequencias[L],PSD[L])
    plt.plot(frequencias,magnitude)
    #return(frequencias,magnitude)


    def tracar_defeitos(defeitos):
        '''
        Função para traçar defeitos nos respetivos locais

        @param: str dados_fea - tuple constituida por par defeito-local
        @return: Plot gráfico dos locais das anomalias 
        '''
        
        anomalias=[] # Inicializar lista de defeitos
        locais=[] # Inicializar lista de locais referentes aos defeitos
        
        # Associar valores às respetivas listas
        for i in range(0,len(defeitos)):
            anomalias.append(defeitos[i][0])
            #locais.append(int(defeitos[i][1])+16.47) # Teste Local
            locais.append(int(defeitos[i][1]))
        
        # tempo=numpy.array(locais)/(3200*0.5)

        # for i in range(0,len(locais)):
        #     ax1.axvline(x=tempo[i], color="r", linewidth="1.5")
        
        for i in range(0,len(locais)):
            ax1.axvline(x=locais[i], color="r", linewidth="1.5")
    
    def tracar_sensores(sensores):
        '''
        Função para traçar defeitos nos respetivos locais

        @param: str dados_fea - tuple constituida por par defeito-local
        @return: Plot gráfico dos locais das anomalias 
        '''
        
        defeitos=[] # Inicializar lista de defeitos
        locais=[] # Inicializar lista de locais referentes aos defeitos
        
        # Associar valores às respetivas listas
        for i in range(0,len(sensores)):
            defeitos.append(sensores[i][0])
            #locais.append(int(sensores[i][1])+16.47) # Teste Local
            locais.append(int(sensores[i][1]))
        
        # tempo=numpy.array(locais)/(3200*0.5)

        # for i in range(0,len(locais)):
        #     ax1.axvline(x=tempo[i], color="r", linewidth="1.5")
        
        for i in range(0,len(locais)):
            ax1.axvline(x=locais[i], color="b", linewidth="1.5")
    
    tracar_defeitos(anomalias)
    tracar_sensores(sensores)

    # Eixos
    ax1.set_xlabel("Time (s)")
    ax2.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")
    #ax2.set_ylabel("PSD (V²/Hz)")
    ax2.set_ylabel("Amplitude (Normalizada)")
    ax1.grid()
    ax2.grid()
    #ax1.set_xlim(0,t[-1])
    ax1.set_xlim(0,distancia[-1])
    ax2.set_xlim(0,1e6)
    plt.show()


def butterworth(sinal):
    N=len(sinal)
    T=5e-7
    fs=1/T
    t=numpy.linspace(0, N*T, N)  # Criar vetor de tempo corretamente

    # Definir os limites do filtro passa-banda
    f_low=40000  # Frequência inferior da banda de passagem (40 kHz)
    f_high=60000  # Frequência superior da banda de passagem (60 kHz)
    ordem=4  # Ordem do filtro

    # Parametros filtro Butterworth
    b,a=signal.butter(ordem, [f_low / (fs / 2), f_high / (fs / 2)], btype='band')

    # Aplicar o filtro
    sinal_filtrado=signal.filtfilt(b, a, sinal)

    # Plot dos sinais
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(t, sinal, label='Sinal com Ruído')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t, sinal_filtrado, label='Sinal Filtrado', color='r')
    plt.legend()
    plt.xlabel('Tempo (s)')
    plt.show()
    return(sinal_filtrado)

def calcular_snr(sinal):
    #https://www.sciencedirect.com/science/article/pii/S0041624X17301622?casa_token=-OEbi_PukRsAAAAA:iNilTlVxpU-LrbMbkTFJRXcMoIPAZRhztP3EOSwyh5SK8bvA69I0tpt3PLzLgFhPibowHwYeLw
    
    '''
    Função para avaliar SNR

    @param: list sinal - lista que contem valores de amplitude do sinal
    @return: float SNR - Valor real do SNR
    ''' 

    S=numpy.max(numpy.abs(numpy.array(sinal)))  # Amplitude Máxima do Sinal
    N=numpy.sqrt(numpy.mean(numpy.array(sinal)**2))  # Nivel de Ruido - RMS do sinal completo

    snr_db = 20 * numpy.log10(S / N)

    return snr_db


  














sinal,dicionario=ler_nano("Teste2.nano") # Na pratica não vou precisar das leituras do segundo canal (y)
anomalias,sensores=ler_fea("Teste.fea")
grafico_fft_anomalias(sinal,anomalias)
grafico(sinal,anomalias)
sinal_filtrado=butterworth(sinal)
print(f"O sinal tem um SNR de: {calcular_snr(sinal):.2f}")
print(f"O sinal tem um SNR de: {calcular_snr(sinal_filtrado):.2f}")
