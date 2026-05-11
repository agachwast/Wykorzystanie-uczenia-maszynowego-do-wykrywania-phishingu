# Wykorzystanie-uczenia-maszynowego-do-wykrywania-phishingu

Przedmiotem projektu jest zastosowanie metod uczenia maszynowego do wykrywania wiadomości SMS o charakterze phishingowym. Badanie obejmuje porównanie różnych metod reprezentacji tekstu oraz ocenę wpływu technik wstępnego przetwarzania danych na skuteczność klasyfikacji.

W badaniu zastosowano metody reprezentacji tekstu, takie jak Bag of Words, TFIDF oraz modele embeddingowe Word2Vec i Doc2Vec, a także ich warianty hybrydowe.
Dodatkowo przeprowadzono eksperymenty z wykorzystaniem różnych technik wstępnego przetwarzania tekstu, obejmujących m.in. standaryzację, tokenizację, 
stemming oraz usuwanie słów stop. Do klasyfikacji wykorzystano algorytm lasu losowego.

Skuteczność modeli oceniono za pomocą miar takich jak dokładność, wartość F1, czułość, precyzja oraz swoistość. Zawarto również
analizę wpływu cech z wykorzystaniem metody SHAP.

W pracy wykorzystano dane pochodzące z zbioru SMS-ów (S. Mishra, D. Soni, SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN
RECOGNITION, Mendeley Data, V1, 2020, doi: 10.17632/f45bkkt8pr.1) zawierającego wiadomości sklasyfikowane jako bezpieczne, spam oraz smishing. 
