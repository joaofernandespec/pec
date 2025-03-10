import codigo
import numpy
sinal,dicionario=codigo.ler_nano("Teste200.nano") # Na pratica não vou precisar das leituras do segundo canal (y)


def sub_bandas(sinal,detalhes_sinal):
    '''
    Função para definir intervalos das subbandas

    @param: list sinal - lista que contem valores de amplitude do sinal
    @param: dicionario detalhes_sinal - dicionario que contem dados do sinal
    @return: tuple f_low,f_high - Tuple formada pelos valores dos minimos e máximos (respetivamente) de cada sub-banda

    ''' 
    frequencia_central=float(detalhes_sinal["Frequency"])*1000 # Frequência central do sinal emitido
    print(frequencia_central)

    #Calculo dos Parâmetros SSP - Valores 607705
    B_largura= 40000 # 90 da Energia # Vou definir como sendo um intervalor de 40000
    frequencia_minima=frequencia_central-B_largura/2 # Frequancia minima
    frequencia_maxima=frequencia_central+B_largura/2 # Frequencia maxima

    Bfilt_Largura_filtro=B_largura/7
    F_separacao_entre_filtros=B_largura/4.5 # F Tem de ser, pelo menos, 4 vezes menores que a largura de cada banda
    N_numero_filtros=int(B_largura/F_separacao_entre_filtros)+1

    # Frequencias centrais de cada Sub-banda
    f_low=[0] # Inicializar lista correspondente as frequencias minimas de cada banda
    f_high=[0] # Inicializar lista correspondente as frequencias maximas de cada banda

    f_low[0]=frequencia_minima-F_separacao_entre_filtros
    f_high[0]=f_low[0]+Bfilt_Largura_filtro

    for i in range(1,N_numero_filtros+1):
        f_low.append(f_low[-1]+F_separacao_entre_filtros)
        f_high.append(f_low[-1]+F_separacao_entre_filtros)

    return f_low,f_high

#f_low,f_high=sub_bandas(sinal,dicionario)

def windows_gausinas(f_low,f_high):
    f_central=[]
    for i in range(0,len(f_low)):
        f_central.append((f_low[i]+f_high[i])/2)


    numpy.exp(-0.5*((x-xc)/sigma)**2)
    w=[]   
    w(x)=
