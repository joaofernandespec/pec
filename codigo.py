import json
import matplotlib.pyplot as plt
import numpy

def ler_nano(ficheiro): #Ainda preciso de dar output a alguns parametros importantes
    '''
    Função para ler valores numéricos de um arquivo .nano e separar os valores em duas listas: x_values e y_values.
    Após isto retorna os valores da amplitude (X) e o dicionário com as características do sinal emitido

    @param: str file - O nome do ficheiro texto .nano
    @return: tuple - Tuple formada por: Valores de X e Dicionario
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
        valores = linha.strip().split('\t') # Formatar string para um float de modo a traçar gráfico

        x_values.append(float(valores[0]))
        y_values.append(float(valores[1]))

    return x_values,dicionario_detalhes # Valores do 2ºCanal(y) não são necessários


def grafico(amplitude, frequencia):
    """
    Cria um gráfico onde X=Distância Percorrida pelo som e Y=Amplitude de Sinal
    
    @param amplitude: Lista com os valores da amplitude (V)
    @param frequencia: Frequência do sinal (Hz)
    """

    # Cálculo do período e distância
    # Criar uma lista de numeros inteiros [1,2,3,4...]
    x=[]
    print(len(amplitude))
    for i in range(0,len(amplitude)):
        x.append(i)
    
    x=numpy.array(x,dtype=int)

    #periodo = 1 / frequencia # Podemos assumir que é 5e-7s
    periodo=5e-7
    velocidade = 3200  # Velocidade do som (m/s) no material
    distancia=velocidade*periodo*x*0.5

    # Plot
    plt.plot(distancia,amplitude)
    plt.xlabel("Distância (m)")
    plt.ylabel("Amplitude (V)")
    plt.title("Sinal Sonoro")
    return plt.show()

def ler_fea(ficheiro):
    '''
    Função para indicar onde se encontram as anomalias do ensaio teste

    @param: str file - O nome do ficheiro texto .fea
    @return: tuple - Tuple formada por duas listas: defeitos e local_defeitos
    '''
    
    dicionario_anomalias={
        "W": "Weld",
        "PS": "Suport",
        "F": "Flange",
        "T": "Tee_Branch",
        "EW": "Elbow",
        "B": "Brace",
        "D": "Defect",
        "I": "Indication",
        "V": "Valves"
    }

    file=open(ficheiro,"r")
    y=file.read()

    posicao=y.find("\"")   # Encontrar a posicao da primeira chaveta """
    posicao=y.find("\"",posicao+1) #Encontrar a posicao da segunda "
    y=y[posicao+2:]
    y=y.split("\n")
    posicao=y.index("")
    y=y[0:posicao]

    result = []
    for item in y:
        name, values = item.split(" = ")
        first_value = values.split("\t")[0].replace('"', '').replace(',', '.')
        result.append((name, float(first_value)))

    return result



x,y=ler_nano("Teste2.nano")
anomalias=ler_fea("Teste.fea")
grafico(x,70000)
