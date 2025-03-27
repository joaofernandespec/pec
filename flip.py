from codigo import ler_nano,ler_fea
import numpy as np
import matplotlib.pyplot as plt
import os

def fft_zona_anomalia(sinal,anomalias,sensor,dicionario):
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
            #print (locais_anomalias_direcao_correta,locais_anomalias_direcao_incorreta)
            return (locais_anomalias_direcao_correta,locais_anomalias_direcao_incorreta)
        
    sinal_direcao_correta,sinal_direcao_oposta=tracar_defeitos(anomalias,sensor,dicionario) # Chamar função
    
    def fft(sinal, locais_anomalias, dt=5e-7, intervalo_m=0.1, num_janelas=10, pontos_por_metro=1250): # Parametros default do sinal, quando nao especificados ele vai assumir estes valores
        
        # Se não tiver nenhuma anomalia o programa não executa
        if not locais_anomalias:
            print("Nenhuma anomalia na direção correta")
            return

        intervalo_pontos=int(intervalo_m*pontos_por_metro)
        fs = 1 / dt  # Frequência de amostragem (2 MHz)

        # Pasta para guardar os gráficos
        pasta_destino = "fft_graficos"

        # De modo a comparar os gráficos vou precisar de os normalizar
        max_magnitude = 0
        for idx,local_anomalia in enumerate(locais_anomalias):
            indice_anomalia=int(local_anomalia * pontos_por_metro)
            for i in range(-num_janelas,num_janelas):
                inicio=indice_anomalia+(i*intervalo_pontos)
                fim=inicio+intervalo_pontos
                if inicio<0 or fim>len(sinal):
                    continue
                segmento=sinal[inicio:fim]
                fft_resultado=np.fft.fft(segmento)
                max_magnitude=max(max_magnitude,np.max(np.abs(fft_resultado[:len(fft_resultado)//2])))


        for idx, local_anomalia in enumerate(locais_anomalias):
            indice_anomalia = int(local_anomalia * pontos_por_metro)

            # Criar 10 janelas antes e 10 depois
            for i in range(-num_janelas,num_janelas):
                inicio=indice_anomalia+(i*intervalo_pontos)
                fim=inicio+intervalo_pontos

                if inicio<0 or fim>len(sinal):  # Garantir que os índices estão dentro do sinal
                    continue

                segmento=sinal[inicio:fim]
                fft_resultado=np.fft.fft(segmento)
                freq=np.fft.fftfreq(len(segmento), d=dt)  # Frequências em Hz

                plt.figure()
                plt.plot(freq[:len(freq)//2], np.abs(fft_resultado)[:len(freq)//2])
                plt.title(f'FFT {i+num_janelas+1}/20 - Intervalo: {inicio/pontos_por_metro:.2f}m a {fim/pontos_por_metro:.2f}m')
                plt.xlabel('Frequência (Hz)')
                plt.ylabel('Magnitude')
                plt.ylim(0, max_magnitude)
               
                # Nome do arquivo para cada gráfico
                nome_arquivo = f"FFT_{idx+1}_{i+num_janelas+1}.png"
                caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)

                # Guardar gráfico na pasta
                plt.savefig(caminho_arquivo, dpi=300)  # dpi=300
                plt.close()  # Fechar o gráfico

    fft(sinal, sinal_direcao_correta)




sinal,dicionario=ler_nano("Teste2.nano") # Na pratica não vou precisar das leituras do segundo canal (y)
anomalias,mot=ler_fea("Teste.fea")
fft_zona_anomalia(sinal,anomalias,mot,dicionario)