import numpy as np
import codigo
import matplotlib.pyplot as plt

import numpy as np

def polarity_threshold_minimization(signals):
    signals = np.real(signals)  # Descarta a parte imaginária
    signals = np.array(signals)  # Garante que os sinais estão em formato NumPy
    output = []  # Lista para armazenar os valores processados

    for i in range(len(signals[0])):  # Iterar sobre cada coluna (tempo)
        minimo = signals[0][i]
        mudanca_sinal = False

        for j in range(1, len(signals)):  # Iterar sobre cada linha (sinais)
            if minimo > 0:
                if signals[j][i] < 0:
                    mudanca_sinal = True
                    break  # Sair do loop ao detectar mudança de sinal
                elif signals[j][i] < minimo:
                    minimo = signals[j][i]
            else:
                if signals[j][i] > 0:
                    mudanca_sinal = True
                    break  # Sair do loop ao detectar mudança de sinal
                elif signals[j][i] > minimo:
                    minimo = signals[j][i]

        if mudanca_sinal:
            output.append(0)
        else:
            output.append(abs(minimo))
    # plt.plot(output)
    # plt.show()
    return output



sinal,dicionario=codigo.ler_nano("Teste2.nano") # Na pratica não vou precisar das leituras do segundo canal (y)
anomalias,mot=codigo.ler_fea("Teste.fea")
sinal_ssp=codigo.grafico_fft_gausian(sinal,dicionario)
sinal_ssp = np.real(sinal_ssp)
output=polarity_threshold_minimization(sinal_ssp)
plt.plot(sinal)
plt.plot(output,color="red")
plt.show()