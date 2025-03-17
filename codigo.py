import json
import math
import matplotlib.pyplot as plt
import numpy
import scipy.signal as signal
from scipy.interpolate import interp1d


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
   
    # A minha ideia inicial era inverter o gráfico para comparar com a imagem do tubo mas não era prático, pelo que comentei esta seccao toda

    # if dicionario_detalhes["Direction"]=="Left": # Se o sinal for emitido para a esquerda temos de inverter o gráfico
    #     x_values.reverse()

    return x_values,dicionario_detalhes # Valores do 2ºCanal(y) não são necessários


def grafico(amplitude,dados_fea):
    """
    Cria um gráfico onde X=Distância Percorrida pelo som e Y=Amplitude de Sinal. Traça, também, os locais das anomalias
    
    @param amplitude: Lista com os valores da amplitude (V)
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


def grafico_fft_anomalias(sinal,anomalias,sensores,dicionario):
    '''
    Função para traçar espectro do domínio temporal e das frequências

    @param: list sinal - lista com valores de amplitude lidos por MOT
    @param: lista anomalias - lista com locais das anomalias
    @param: lista sensores - lista com locais dos sensores
    @param: dicionario dicionario - dicionario com as caracteristicas do sinal

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
    frequencias=numpy.fft.fftfreq(N,T) # Vetor frequencias
    magnitude = numpy.abs(fourier) / max(numpy.abs(fourier)) # Normalizar
    
    #Inicialmente ia utilizar a densidade espectral de potencia para quantificar a FFT mas depois nao se tornou muito prático
    #PSD=(numpy.abs(fourier)**2)/(N*T) # Densidade Espectral de Potência
    #L=numpy.arange(0,numpy.floor(N/2),dtype='int') # So queremos metade do espectro

    # Plotar grafico
    #plt.plot(t,sinal) # Caso queiramos colocar eixo segundo o tempo
    fig,[ax1,ax2]=plt.subplots(nrows=2,ncols=1)
    plt.sca(ax1)
    plt.plot(distancia,sinal)
    plt.sca(ax2)
    plt.plot(frequencias,magnitude)
    #plt.plot(frequencias[L],PSD[L])
    #return(frequencias,magnitude)


    def tracar_defeitos(defeitos,mot,dicionario):
        '''
        Função para traçar defeitos nos respetivos locais

        @param: list defeitos - lista de tuples constituidas por par defeito-local
        @param: list mot - lista de tuples constituidas por par sensor-local
        @param: dicionario dicionario - dicionario com as caracteristicas do sinal


        @return: Plot gráfico dos locais das anomalias 
        '''

        locais_anomalias_direcao_correta=[] # Inicializar lista de locais_anomalias referentes as anomalias à esquerda do sensor
        locais_anomalias_direcao_incorreta=[] # Inicializar lista de locais_anomalias referentes as anomalias à direita do sensor
        locais_sensores=[]
        
        for i in range(0,len(mot)):
            locais_sensores.append(float(mot[i][1])) # Dar .append aos locais dos sensores
           
        # Associar valores às respetivas listas
        # Se o sinal for emitido para a direita e o defeito se encotrar à direita este vai ser projetado com 80% da Energia do Sinal - Cor vermelha
        # Se o sinal for emitido para a esquerda e o defeito se encotrar à esquerda este vai ser projetado com 80% da Energia do Sinal - Cor vermelha
        # Se o sinal for emitido numa direcao contraria à posicao do sinal entao este vai ser apenas projetado com 20% da Energia do Sinal pelo que vou marcar a uma cor diferente - Cor Laranja

        for i in range(0,len(defeitos)):
            if (defeitos[i][1]>locais_sensores[0] and dicionario["Direction"]=="Right") or (defeitos[i][1]<locais_sensores[0] and dicionario["Direction"]=="Left"):
                locais_anomalias_direcao_correta.append(float(defeitos[i][1])) # Dar .append aos locais das anomalias consoante direcao do sinal
            else:
                locais_anomalias_direcao_incorreta.append(float(defeitos[i][1])) # Dar .append aos locais das anomalias consoante direcao do sinal

        # tempo=numpy.array(locais_anomalias)/(3200*0.5) # Caso queiramos utilizar o eixo do tempo - Mesmo raciocínio

        # for i in range(0,len(locais_anomalias)):
        #     ax1.axvline(x=tempo[i], color="r", linewidth="1.5")
        
        for i in range(0,len(locais_anomalias_direcao_correta)):
            ax1.axvline(x=(abs(locais_anomalias_direcao_correta[i]-locais_sensores[0])), color="r", linewidth="1.5") # Traçar retas verticais consoante posicao relativa ao sensor

        for i in range(0,len(locais_anomalias_direcao_incorreta)):
            ax1.axvline(x=(abs(locais_anomalias_direcao_incorreta[i]-locais_sensores[0])), color="orange", linewidth="1.5") # Traçar retas verticais consoante posicao relativa ao sensor
    
    tracar_defeitos(anomalias,sensores,dicionario) # Chamar função

    # Eixos
    ax1.set_xlabel("Distância (m)")
    ax2.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Amplitude")
    ax2.set_ylabel("Amplitude (Normalizada)")
    ax1.grid()
    ax2.grid()
    #ax1.set_xlim(0,t[-1])
    #ax2.set_ylabel("PSD (V²/Hz)")
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

def grafico_fft_gausian(sinal,dicionario):
    # Computar sinal
    N=len(sinal)  # Número de pontos
    T=5e-7  # Período de amostragem
    fs=1/T # Frequência de amostragem
    t=numpy.linspace(0, N*T, N)  # Criar vetor de tempo
    #distancia=t*3200*0.5

    def sub_bandas(detalhes_sinal):
        '''
        Função para definir intervalos das subbandas

        @param: dicionario detalhes_sinal - dicionario que contem dados do sinal
        @return: tuple lista_frequencias,amplitudes,N_numero_filtros - Tuple formada por lista de frequencias, amplitude para cada frequencia e Nº de filtros respetivamente

        ''' 
        frequencia_central=float(detalhes_sinal["Frequency"])*1000 # Frequência central do sinal emitido

        #Calculo dos Parâmetros SSP - Valores 607705
        B_largura= 100000 # 90 da Energia # Vou definir como sendo um intervalor de 50000
        frequencia_minima=frequencia_central-B_largura/2 # Frequancia minima
        frequencia_maxima=frequencia_central+B_largura/2 # Frequencia maxima

        # Parametros utilizados
        Bfilt_Largura_filtro=B_largura/7
        F_separacao_entre_filtros=Bfilt_Largura_filtro/4.5
        N_numero_filtros=math.ceil(B_largura/F_separacao_entre_filtros)+1

        # Intervalo de frequencias de cada Sub-banda
        f_low=[0] # Inicializar lista correspondente as frequencias minimas de cada banda
        f_high=[0] # Inicializar lista correspondente as frequencias maximas de cada banda

        f_low[0]=frequencia_minima-F_separacao_entre_filtros
        f_high[0]=f_low[0]+Bfilt_Largura_filtro

        for i in range(1,N_numero_filtros+1):
            f_low.append(f_low[-1]+F_separacao_entre_filtros)   
            f_high.append(f_low[-1]+Bfilt_Largura_filtro)

        # Inicializar listas
        f_central=[] # Lista correspondente aos valores centrais de cada distribuicao gaussiana
        lista_frequencias=[] # Lista que sera composta por todas as listas correspondentes as distribuicaos gausianas
        amplitudes=[] # Lista a que serão alocadas as amplitudes do sinal

        fwhm=5000 
        sigma=fwhm/(2*math.sqrt((2*math.log(2)))) # Calculo do Sigma com base no fwhm

        for i in range(0,len(f_low)):
            f_central.append((f_low[i]+f_high[i])/2) # Calculo da frequencia central
            freq_range=numpy.linspace(f_low[i],f_high[i],1000) # Lista com pontos espacados igualmente - Util para tracar grafico
            lista_frequencias.append(freq_range)

            # Calculo da Gaussiana
            gaussiana=numpy.exp(-0.5*(((numpy.array(lista_frequencias[0])-f_central[0]))/sigma)**2)
            amplitudes.append(gaussiana)
        return lista_frequencias,amplitudes,N_numero_filtros,f_low,f_high

    lista_frequencias,amplitudes,N_numero_filtros,f_low,f_high=sub_bandas(dicionario)

    fourier=numpy.fft.fft(sinal,N)
    frequencias=numpy.fft.fftfreq(N,T)

    mask = (frequencias >= 10000) & (frequencias <= 110000)
    fourier_filtrado = fourier[mask]
    frequencias_filtradas = frequencias[mask]

    magnitude = numpy.abs(fourier_filtrado) / max(numpy.abs(fourier_filtrado))
    plt.plot(frequencias_filtradas,magnitude)

    cores=["red","orange","yellow","blue","green","purple"]
    p=0

    for i in range(0,N_numero_filtros):
        if p<=5:
            plt.plot(lista_frequencias[i],amplitudes[i],color=cores[p],linewidth=2)
            p+=1
        else:
            p=0
            plt.plot(lista_frequencias[i],amplitudes[i],color=cores[p],linewidth=2)
            p+=1
    #plt.show()

    def filtrar_pontos(x_values1, y_values1, x_values2, y_values2):
        # Criar uma função de interpolação para y_values2 nos pontos de x_values2
        interp_y2 = interp1d(x_values2, y_values2, kind='linear', fill_value="extrapolate")

        # Interpolar os valores de y_values2 nos pontos de x_values1
        y2_interp_values = interp_y2(x_values1)

        # Filtrar os pontos onde y1 < y2_interpolado
        x_filtrado = [x for x, y1, y2 in zip(x_values1, y_values1, y2_interp_values) if y1 < y2]
        y_filtrado = [y1 for y1, y2 in zip(y_values1, y2_interp_values) if y1 < y2]

        return x_filtrado, y_filtrado

    x=[0] * N  # Cria uma lista com N elementos, todos iguais a 0
    y=[0] * N
    sinal_recuperado=numpy.zeros((N_numero_filtros, N), dtype=complex)
    for i in range(0,N_numero_filtros):
        x[i],y[i]=filtrar_pontos(frequencias_filtradas,magnitude,lista_frequencias[i],amplitudes[i])

        #plt.plot(x[7],y[7],color="brown")
        plt.show()

        # **Reconstruir a FFT modificada**
        fourier_modificado = numpy.zeros_like(fourier, dtype=complex)  # Criar FFT zerada
        fourier_modificado[mask] = fourier[mask] * (numpy.array([y[i] if f in x[i] else 0 for f, y[i] in zip(frequencias_filtradas, magnitude)]))  # Manter apenas os valores filtrados

        # **Aplicar IFFT**
        sinal_recuperado[i] = numpy.fft.ifft(fourier_modificado)
    
    soma=[0]*len(sinal_recuperado)
    for i in range(0,len(sinal_recuperado)):
        sinal_recuperado[i]=sinal_recuperado[i]/numpy.max(numpy.real(sinal_recuperado[i]))
   
    fig, axes = plt.subplots(5, 6, figsize=(18, 10))
    for i, ax in enumerate(axes.flat):
        ax.plot(t, sinal_recuperado[i])  # Desenha o gráfico do sinal recuperado[i]
        ax.set_title(f'Gráfico {i+1}')  # Título do subgráfico
        ax.grid(True)
        ax.set_xlabel('Tempo (t)')
        ax.set_ylabel('Sinal Recuperado')
    #plt.plot(t, numpy.real(sinal_recuperado[15]))
   
    #plt.title("Sinal Recuperado após Filtro")
    plt.show()
    return sinal_recuperado


  














sinal,dicionario=ler_nano("Teste2.nano") # Na pratica não vou precisar das leituras do segundo canal (y)
anomalias,mot=ler_fea("Teste.fea")
grafico_fft_anomalias(sinal,anomalias,mot,dicionario)

##grafico(sinal,anomalias) # Em principio não vou utilizar
sinal_filtrado=butterworth(sinal)
print(f"O sinal tem um SNR de: {calcular_snr(sinal):.2f}")
print(f"O sinal tem um SNR de: {calcular_snr(sinal_filtrado):.2f}")
sinal_ssp=grafico_fft_gausian(sinal,dicionario)


#grafico_fft_anomalias(sinal_ssp,anomalias,mot,dicionario)
