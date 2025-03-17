import codigo
import numpy
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
    sinal_recuperado=[0] * N
    for i in range(0,N_numero_filtros):
        x[i],y[i]=filtrar_pontos(frequencias_filtradas,magnitude,lista_frequencias[i],amplitudes[i])

        #plt.plot(x[7],y[7],color="brown")
        plt.show()

        # **Reconstruir a FFT modificada**
        fourier_modificado = numpy.zeros_like(fourier, dtype=complex)  # Criar FFT zerada
        fourier_modificado[mask] = fourier[mask] * (numpy.array([y[i] if f in x[i] else 0 for f, y[i] in zip(frequencias_filtradas, magnitude)]))  # Manter apenas os valores filtrados

        # **Aplicar IFFT**
        sinal_recuperado[i] = numpy.fft.ifft(fourier_modificado)
    
    for i in range(0,len(sinal_recuperado)):
        sinal_recuperado[i]=sinal_recuperado[i]/numpy.max(numpy.real(sinal_recuperado[i]))
        
    # **Plot do sinal recuperado**
    plt.plot(t,numpy.real(sinal_recuperado[8]))  # Apenas a parte real do sinal

    plt.title("Sinal Recuperado após Filtro")
    plt.show()

grafico_fft_gausian(x,dicionario)