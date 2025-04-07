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
        i=0
        name, values=item.split(" = ")
        first_value=values.split("\t")[0].replace('"', '').replace(',', '.')

        if name==dicionario["Tags"][i]:
            u=i

        if "_" in name: # Esta condição, em principio, vai se verificar sempre - Medida contra possíveis erros
            posicao_=name.find("_") # Encontrar _ que separa tipo de defeito e nº daquele tipo de defeito
            
            # Trabalhar string
            key=name[0:posicao_]
            tipo_anomalia=dicionario_anomalias[key] # Traduz key-valor
            name=tipo_anomalia
            sensores.append((name, float(first_value))) # Anexar ao result que irá ser retornado
        i+=1
    return result,sensores,u


def grafico_fft_anomalias(sinal,anomalias,sensores,dicionario,ger):
    '''
    Função para traçar espectro do domínio temporal e das frequências; Traça tambem as anomalias/sensores

    @param: list sinal - lista com valores de amplitude lidos por MOT
    @param: lista anomalias - lista com locais das anomalias
    @param: lista sensores - lista com locais dos sensores
    @param: dicionario dicionario - dicionario com as caracteristicas do sinal

    @return: None - Gráficos com espectros do domínio temporal e das frequências
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


    def tracar_defeitos(defeitos,mot,dicionario,ger):
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
        
        u=0 # Vai funcionar como um Localizador
        for i in range(0,len(mot)):
            locais_sensores.append(float(mot[i][1])) # Dar .append aos locais dos sensores
           
        # Associar valores às respetivas listas
        # Se o sinal for emitido para a direita e o defeito se encotrar à direita este vai ser projetado com 80% da Energia do Sinal - Cor vermelha
        # Se o sinal for emitido para a esquerda e o defeito se encotrar à esquerda este vai ser projetado com 80% da Energia do Sinal - Cor vermelha
        # Se o sinal for emitido numa direcao contraria à posicao do sinal entao este vai ser apenas projetado com 20% da Energia do Sinal pelo que vou marcar a uma cor diferente - Cor Laranja

        for i in range(0,len(defeitos)):
            if (defeitos[i][1]>locais_sensores[ger] and dicionario["Direction"]=="Right") or (defeitos[i][1]<locais_sensores[ger] and dicionario["Direction"]=="Left"):
                locais_anomalias_direcao_correta.append(float(defeitos[i][1])) # Dar .append aos locais das anomalias consoante direcao do sinal
            else:
                locais_anomalias_direcao_incorreta.append(float(defeitos[i][1])) # Dar .append aos locais das anomalias consoante direcao do sinal

        # tempo=numpy.array(locais_anomalias)/(3200*0.5) # Caso queiramos utilizar o eixo do tempo - Mesmo raciocínio

        # for i in range(0,len(locais_anomalias)):
        #     ax1.axvline(x=tempo[i], color="r", linewidth="1.5")
        
        for i in range(0,len(locais_anomalias_direcao_correta)):
            ax1.axvline(x=(abs(locais_anomalias_direcao_correta[i]-locais_sensores[ger])), color="r", linewidth="1.5") # Traçar retas verticais consoante posicao relativa ao sensor

        for i in range(0,len(locais_anomalias_direcao_incorreta)):
            ax1.axvline(x=(abs(locais_anomalias_direcao_incorreta[i]-locais_sensores[ger])), color="orange", linewidth="1.5") # Traçar retas verticais consoante posicao relativa ao sensor
    
    tracar_defeitos(anomalias,sensores,dicionario,ger) # Chamar função

    # Eixos
    ax1.set_xlabel("Distância (m)")
    ax2.set_xlabel("Frequencia (Hz)")
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
    '''
    Função para filtrar parte do sinal sonoro

    @param: list sinal - Lista de valores da amplitude lidos pelo sensor MOT
    @return: list sinal_filtrado - Sinal filtrado pelo filtro butterworth de 4a ordem
    '''
    N=len(sinal) # Numero de valores lidos pelo MOT
    T=5e-7 # Periodo de amostragem
    fs=1/T # Frequência de amostragem
    t=numpy.linspace(0, N*T, N)  # Criar vetor de tempo corretamente

    # Definir os limites do filtro passa-banda
    f_low=40000  # Frequência inferior da banda de passagem (40 kHz)
    f_high=60000  # Frequência superior da banda de passagem (60 kHz)
    ordem=4  # Ordem do filtro

    # Parametros filtro Butterworth
    b,a=signal.butter(ordem,[f_low/(fs/2),f_high/(fs/2)],btype='band')

    # Aplicar o filtro
    sinal_filtrado=signal.filtfilt(b,a,sinal)

    # Plot dol sinal
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
    @return: float snr_db - Valor real do SNR
    ''' 

    S=numpy.max(numpy.abs(numpy.array(sinal)))  # Amplitude Máxima do Sinal
    N=numpy.sqrt(numpy.mean(numpy.array(sinal)**2))  # Nivel de Ruido - RMS do sinal completo

    snr_db=20*numpy.log10(S/N) # Calculo do SNR

    return snr_db

def grafico_fft_gausian(sinal,dicionario):
    '''
    Função para realizar a transformada de fourier de cada janela gaussiana definida pela funcao (sub_bandas)

    @param: list sinal - Lista que contem valores de amplitude do sinal
    @return: dicionario dicionario - dicionario que contem informaçóes a cerca do sinal
    ''' 
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
        B_largura= 40000 # 90% da Energia # Vou definir como sendo um intervalor de 1000000
        frequencia_minima=frequencia_central-B_largura/2 # Frequancia minima
        frequencia_maxima=frequencia_central+B_largura/2 # Frequencia maxima

        # Parametros utilizados
        Bfilt_Largura_filtro=B_largura/3 # Largura de Sub-banda
        F_separacao_entre_filtros=Bfilt_Largura_filtro/4.5 # Separacao entre os filtros
        N_numero_filtros=math.ceil(B_largura/F_separacao_entre_filtros)+1 # Numero de filtros

        # Intervalo de frequencias de cada Sub-banda
        f_low=[0] # Inicializar lista correspondente as frequencias minimas de cada banda
        f_high=[0] # Inicializar lista correspondente as frequencias maximas de cada banda

        # Vou estabelecer o intervalo da primeira sub_banda e depois inicializar um loop para criar as restantes e anexar a f_low e f_high
        f_low[0]=frequencia_minima-F_separacao_entre_filtros # Menor frequencia da sub-banda de menor frequencia
        f_high[0]=f_low[0]+Bfilt_Largura_filtro # Maior frequencia da sub_banda de maior frequencia

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
            lista_frequencias.append(freq_range) # Lista onde irão estar alocados as listas formadas pelas abcissas das sub-bandas

            # Calculo da Gaussiana
            gaussiana=numpy.exp(-0.5*(((numpy.array(lista_frequencias[0])-f_central[0]))/sigma)**2)
            amplitudes.append(gaussiana) # Lista onde irão estar alocados as listadas formadas pelas ordenadas das sub-bandas
        return lista_frequencias,amplitudes,N_numero_filtros,f_low,f_high,B_largura

    lista_frequencias,amplitudes,N_numero_filtros,f_low,f_high,B_largura=sub_bandas(dicionario) # Executação da função

    fourier=numpy.fft.fft(sinal,N) # Aplicada da transformada de Fourier
    frequencias=numpy.fft.fftfreq(N,T) # Criar uma lista formada pelas frequencias (De modo a construir um gráfico)

    mask = (frequencias >= dicionario["Frequency"]*1000-B_largura/2) & (frequencias <= dicionario["Frequency"]*1000+B_largura/2) # Definição de um filtro - vamos apenas analisar uma gama de frequencias de 100000
    
    # Aplicar filtros à trnsformada de Fourier
    fourier_filtrado = fourier[mask]
    frequencias_filtradas = frequencias[mask]

    # Definicao do vetor magnitude
    magnitude = numpy.abs(fourier_filtrado) / max(numpy.abs(fourier_filtrado))
    plt.plot(frequencias_filtradas,magnitude) # Plot do gráfico (Ainda sem sub-bandas)

    # Vou agora criar um loop para tracar as sub-bandas
    cores=["red","orange","yellow","blue","green","purple"] # Definicao de um vetor de cores
    p=0 # Contador

    for i in range(0,N_numero_filtros):
        if p<=5:
            plt.plot(lista_frequencias[i],amplitudes[i],color=cores[p],linewidth=2)
            p+=1
        else:
            p=0
            plt.plot(lista_frequencias[i],amplitudes[i],color=cores[p],linewidth=2)
            p+=1
    plt.show()

    def filtrar_pontos(x_values1,y_values1,x_values2,y_values2):
        # Criar uma função de interpolação para y_values2 nos pontos de x_values2
        interp_y2=interp1d(x_values2,y_values2,kind='linear',fill_value="extrapolate")

        # Interpolar os valores de y_values2 nos pontos de x_values1
        y2_interp_values=interp_y2(x_values1)

        # Filtrar os pontos onde y1 < y2_interpolado
        x_filtrado=[x for x,y1,y2 in zip(x_values1,y_values1,y2_interp_values) if y1<y2]
        y_filtrado=[y1 for y1,y2 in zip(y_values1,y2_interp_values) if y1<y2]

        return x_filtrado,y_filtrado
    
    # Para testar o código
    # u,v=filtrar_pontos(frequencias_filtradas,magnitude,lista_frequencias[7],amplitudes[7])
    # plt.plot(u,v,color="brown")
    # plt.plot(lista_frequencias[7],amplitudes[7])
    # plt.show()

    x=[0]*N_numero_filtros # Cria uma lista com N elementos, todos iguais a 0
    y=[0]*N_numero_filtros # Cria um lista de N elementos, todos iguais a 0

    sinal_recuperado=numpy.zeros((N_numero_filtros, N),dtype=complex)
    for i in range(0,N_numero_filtros):
        x[i],y[i]=filtrar_pontos(frequencias_filtradas,magnitude,lista_frequencias[i],amplitudes[i])


        # Reconstruir a FFT modificada
        fourier_modificado=numpy.zeros_like(fourier, dtype=complex)  # Criar FFT zerada
        fourier_modificado[mask]=fourier[mask]*(numpy.array([y[i] if f in x[i] else 0 for f, y[i] in zip(frequencias_filtradas, magnitude)]))  # Manter apenas os valores filtrados

        # Aplicar IFFT
        sinal_recuperado[i]=numpy.fft.ifft(fourier_modificado)

    # Normlizar sinais obtidos
    for i in range(0,len(sinal_recuperado)):
        sinal_recuperado[i]=sinal_recuperado[i]/numpy.max(numpy.real(sinal_recuperado[i]))
   
   # Plotar Gráfico
    fig,axes=plt.subplots(5, 6, figsize=(20, 12))
    for i,ax in enumerate(axes.flat):
        if i+1>len(sinal_recuperado)-1:
            break
        else:
            ax.plot(t, sinal_recuperado[i])  # Desenha o gráfico do sinal recuperado[i]
            ax.grid(True)
    #plt.plot(t, numpy.real(sinal_recuperado[15])) - Apenas para testar codigo
   
    plt.show()
    return sinal_recuperado

def polarity_threshold_minimization(signals):
    signals=numpy.real(signals)  # Descarta a parte imaginária
    signals=numpy.array(signals)  # Garante formato correto
    output=[]  # Lista para armazenar os valores processados

    for i in range(len(signals[0])):  # Iterar sobre cada coluna janela gaussiana
        minimo=signals[0][i] # Definir o minimo inicialmente como o primeiro valor de cada gaussiana
        mudanca_sinal=False

        for j in range(1,len(signals)):  # Iterar sobre cada sinal
            if minimo>0:
                if signals[j][i]<0:
                    mudanca_sinal=True
                    break  # Sair do loop ao detectar mudança de sinal
                elif signals[j][i]<minimo:
                    minimo=signals[j][i]
            else:
                if signals[j][i] > 0:
                    mudanca_sinal=True
                    break  # Sair do loop ao detectar mudança de sinal
                elif signals[j][i] > minimo:
                    minimo=signals[j][i]

        if mudanca_sinal:
            output.append(0)
        else:
            output.append(abs(minimo))
    # plt.plot(output)
    # plt.show()
    return output

def sinal_anomalias_sinal_output(sinal,sinal2,anomalias,mot,dicionario,output,output2):
    # Codigo muito semelhante ao grafico_fft. Nao vou comentar
    N=len(sinal)
    N2=len(sinal2)
    T=5e-7
    fs=1/T
    t=numpy.linspace(0, N*T, N)
    t2=numpy.linspace(0, N2*T, N2)

    distancia=t*3200*0.5
    distancia2=t2*3200*0.5

    fig,[ax1,ax2]=plt.subplots(nrows=2,ncols=1)
    plt.sca(ax1)
    plt.plot(distancia2,sinal2)
    plt.sca(ax2)
    plt.plot(distancia,output,color="green")
    plt.plot(distancia2,output2,color="red")



    def tracar_defeitos(defeitos,mot,dicionario):
        locais_anomalias_direcao_correta=[]
        locais_anomalias_direcao_incorreta=[] 
        locais_sensores=[]
        
        for i in range(0,len(mot)):
            locais_sensores.append(float(mot[i][1]))

        for i in range(0,len(defeitos)):
            if (defeitos[i][1]>locais_sensores[0] and dicionario["Direction"]=="Right") or (defeitos[i][1]<locais_sensores[0] and dicionario["Direction"]=="Left"):
                locais_anomalias_direcao_correta.append(float(defeitos[i][1]))
            else:
                locais_anomalias_direcao_incorreta.append(float(defeitos[i][1]))

        for i in range(0,len(locais_anomalias_direcao_correta)):
            ax1.axvline(x=(abs(locais_anomalias_direcao_correta[i]-locais_sensores[0])), color="r", linewidth="1.5")

        for i in range(0,len(locais_anomalias_direcao_incorreta)):
            ax1.axvline(x=(abs(locais_anomalias_direcao_incorreta[i]-locais_sensores[0])), color="orange", linewidth="1.5") 
    tracar_defeitos(anomalias,mot,dicionario)

    # Eixos
    ax1.set_xlabel("Distância (m)")
    ax2.set_xlabel("Frequencia (Hz)")
    ax1.set_ylabel("Amplitude")
    ax2.set_ylabel("Amplitude (Normalizada)")
    ax1.grid()
    ax2.grid()
    #ax1.set_xlim(0,t[-1])
    #ax2.set_ylabel("PSD (V²/Hz)")
    ax1.set_xlim(0,distancia[-1])
    ax2.set_xlim(0,distancia[-1])
    plt.show()














# «sinal2,dicionario=ler_nano("Teste2.nano") # Na pratica não vou precisar das leituras do segundo canal (y)
# sinal_ssp2=grafico_fft_gausian(sinal2,dicionario)
# sinal_ssp2=numpy.real(sinal_ssp2)
# output2=polarity_threshold_minimization(sinal_ssp2)


sinal,dicionario=ler_nano("Teste1L.nano") # Na pratica não vou precisar das leituras do segundo canal (y)
print(len(sinal))
anomalias,mot,ger=ler_fea("Teste1.fea")
grafico_fft_anomalias(sinal,anomalias,mot,dicionario,ger)
sinal_ssp=grafico_fft_gausian(sinal,dicionario)
sinal_ssp=numpy.real(sinal_ssp)
output=polarity_threshold_minimization(sinal_ssp)
#sinal_anomalias_sinal_output(sinal,sinal2,anomalias,mot,dicionario,output,output2)


sinal2,dicionario=ler_nano("Teste1R.nano") # Na pratica não vou precisar das leituras do segundo canal (y)
sinal_ssp2=grafico_fft_gausian(sinal2,dicionario)
sinal_ssp2=numpy.real(sinal_ssp2)
output2=polarity_threshold_minimization(sinal_ssp2)
sinal_anomalias_sinal_output(sinal,sinal2,anomalias,mot,dicionario,output,output2)

# t=numpy.linspace(0, len(sinal)*5e-7, len(sinal))  # Criar vetor de tempo
# distancia=t*3200*0.5
# plt.plot(distancia,output)
# plt.show()


##grafico(sinal,anomalias) # Em principio não vou utilizar
#sinal_filtrado=butterworth(sinal)
#print(f"O sinal tem um SNR de: {calcular_snr(sinal):.2f}")
#print(f"O sinal tem um SNR de: {calcular_snr(sinal_filtrado):.2f}")

# Vou utilizar estes parametros por agora. Se depois quiser mudar apenas preciso de alterar o valor ou de B ou de Bfilt

