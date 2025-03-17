import codigo
import numpy
import math
import matplotlib.pyplot as plt
x,dicionario=codigo.ler_nano("Teste200.nano")


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

        fwhm=5000 # Como não sei como calcular vou utilizar 10000
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

    diferenca=[] # Vetor onde vão ser alocados a diferencas entre um dado valor da frequencia e todos os pontos da window gausiana
    # Vou fazer uma interpolcao entre os dois pontos mais proximos - Onde a diferenca é minima
    amplitude_interpolacao=[] # Vetor com os valores da amplitude interpolada
    copia_diferenca=[] # Copia do vetor diferenca
    valores_minimos=[]
    # frequencias_filtradas2= Nova lista das abcissas da transformada de fourier
    # magnitude2= lista das ordenadas da transformada de fourier
    for i in range(0, len(f_low)):
        mask=(frequencias_filtradas >= (f_low[i])) & (frequencias_filtradas <= (f_high[i])) # So preciso de procurar entre as frequencias limite da window gaussiana
        frequencias_filtradas2=frequencias_filtradas[mask] # Filtro
        magnitude2=magnitude[mask] # Filtro - Preciso deste vetor para comparar à interpolacao

        if len(frequencias_filtradas2)!=0:
            for i in range(0,len(frequencias_filtradas2)): # Para todos as abcissas da transformada de fourier
                diferenca=[]
                contador=-1
                min1=1000
                for p in range(0,len(lista_frequencias)): # Vou fazer a diferenca entre todos os pontos da window_gaussiana
                    diferenca.append(frequencias_filtradas2[i]-lista_frequencias[p])
                    #print(diferenca)
                diferenca=numpy.array(diferenca)
                copia_diferenca=diferenca.copy()
                copia_diferenca=numpy.array(copia_diferenca)
                for sublista in diferenca:
                    contador+=1
                    sublista = numpy.array(sublista)
                    indeice_central=len(sublista)//2
                    valor_central=(f_low[contador]+f_high[contador])/2
                    sublista = numpy.sort(numpy.abs(sublista))
                    #sublista.sort() # ordeno a diferenca com o intuito de saber os menores valores e, como tal, encontrar os valores mais proximos
                    if sublista[0]<min1:
                        min1 = float(sublista[0])  # Primeiro menor valor
                        min2 = float(sublista[1])  # Segundo menor valor
                        valores_minimos=(min1,min2)
                ordenadas=numpy.exp(-0.5*(((numpy.array(valores_minimos)))/2122.976582960091)**2)
                x=float(ordenadas[0])
                y=float(ordenadas[1])
                amplitude_interpolacao.append((x+y)/2)
        print(amplitude_interpolacao)
            


    for i in range(0,N_numero_filtros):
        if p<=5:
            plt.plot(lista_frequencias[i],amplitudes[i],color=cores[p],linewidth=2)
            p+=1
        else:
            p=0
    plt.show()

    # Agora vou verificar se, para cada ponto, a frequencia esta abaixo ou acima do valor estabelecido pela window Gaussiana
    # Preciso encontrar as abcissas mais proximas para cada valor da frequencia do meu sinal e fazer algum tipo de interpolação
    
    
#lista_frequencias,amplitude=sub_bandas(dicionario)
grafico_fft_gausian(x,dicionario)