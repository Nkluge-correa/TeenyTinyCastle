# Carta Modelo RAIES

**Cartas de modelo são pequenos documentos que acompanham modelos desenvolvidos por aprendizagem de maquina. Tais cartas fornecem detalhes e características de desempenho de um modelo em questão, ajudando a visibilizar condutas de transparência e responsabilização no desenvolvimento de sistemas autônomos/inteligentes.**

**Este formulário foi criado para alimentar um "_template_" padrão de carta modelo, desenvolvido pela Rede de Inteligência Artificial Ética e Segura ([RAIES](https://www.raies.org/)).**

**Para mais informações, contate-nos em [raies@raies.org](mailto:raies@raies.org).**

## DETALHES DO MODELO

1. Este modelo foi desenvolvido por Nicholas Kluge Corrêa, pesquisador da Pontifícia Universidade Católica do Rio Grande do Sul (PUC-RS), Brasil, em agosto de 2022.

2. Este modelo trata-se de um regressor logístico treinado para solucionar uma tarefa de classificação binária. As (duas) possíveis classes representam o risco em se abrir uma linha de crédito para um determinado indivíduo (classe favorecida = aprovado, classe desfavorecida = não aprovado).

3. Este modelo foi desenvolvido apenas por motivações acadêmicas, com o intuito de explorar como diferentes métricas de "fairness" podem ser medidas.

4. O conjunto de dados utilizado é o Credit Approval Data Set, disponibilizado pela UCI Machine Learning Repository. Disponível em: [https://archive.ics.uci.edu/ml/datasets/credit+approval](https://archive.ics.uci.edu/ml/datasets/credit+approval). Este conjunto de dados contém amostras rotuladas (Aprovado/Não Aprovado) de requisições de cartões de crédito.

5. Este conjunto de dados foi escolhido por sua disponibilidade pública.

6. Como a nomenclatura das "features" do Credit Approval Data Set foram mascaradas para preservar a identidade das amostras, as seguintes "features" foram inferidas (a fim de auxiliar na investigação deste classificador): "[Gender", "Age", "Debt", "Married", "Bank Client", "Education", "Race", "Years Employed, "Prior Default", "Employed", "Credit", "Driver’s License", "Citizenship", "Postal Code", "Income", "Approval Status"]. Amostras com valores ausentes tiveram tais valores substituídos pelo respectivo valor médio/moda de cada "feature".

7. Código Aberto

8. Disponível em: [Teeny-Tiny Castle](https://github.com/Nkluge-correa/teeny-tiny_castle).

9. Publicação: nan.

10. Licença: MIT License.

Contato: [nicholas@airespucrs.org](mailto:nicholas@airespucrs.org).

### Citar como

```MarkDown

"@misc{teenytinycastle,
doi = {10.5281/zenodo.7112065},
url = {https://github.com/Nkluge-correa/teeny-tiny_castle},
author = {Nicholas Kluge Corr{\^e}a},
title = {Teeny-Tiny Castle},
year = {2022},
publisher = {GitHub},
journal = {GitHub repository},
note = {Last updated 14 October 2022},
}"

```

## USO PRETENDIDO

1. O uso pretendido para este modelo e o código compartilhado está em apresentar, ao desenvolvedor interessado, ferramentas para se explorar conjuntos de dados e avaliar possíveis riscos associados a utilização de modelos desenvolvidos por aprendizagem de máquina através de dados tendenciosos. Este modelo não foi desenvolvido para ser utilizado em aplicações reais que envolvam a classificação de sujeitos.

2. Este modelo foi desenvolvido para o público acadêmico, cientistas de dados, desenvolvedores e engenheiros de aprendizagem de máquina.

## FATORES

1. O modelo toma como entrada um vetor de "features" que contém duas características (“Gender” e “Prior Default”).

2. Os dados utilizados no desenvolvimento deste modelo possuem uma distribuição tendenciosa no que diz respeito ao atributo gênero. Aproximadamente 70% das amostras possuem o atributo sensível "Masculino", e aproximadamente 68% das amostras que receberam uma classificação positiva ("Aprovado") possuem o atributo sensível "Masculino".

## MÉTRICAS

1. A métrica de performance utilizadas foi de acurácia, onde o modelo alcançou 0.83 (83.33%). A maior parte das classificações erradas feitas por este modelo são Falsos Positivos (amostras que pertencem a classe "Aprovado" sendo classificados como "Não Aprovado").

2. A performance do modelo varia consideravelmente quando o avaliamos em diferentes subgrupos de atributos sensíveis ("Gênero").

## DECLARAÇÃO DE EQUIDADE

1. Este modelo foi avaliado com uma série de métricas de "fairness". Sendo elas: Statistical Parity Ratio, Equal Opportunity Ratio, Predictive Parity Ratio, Predictive Equality Ratio e Accuracy Equality Ratio. Se utilizarmos como medida de corte a regra dos 80% (a razão entre a classificação para a classe benéfica entre grupos privilegiados versus não-privilegiados deve ser menor igual a 80%). O modelo gerado não satisfaz tal condição para a métrica Statistical Parity. O modelo possui uma razão próxima a 1 (0.98) quando avaliado pela métrica Equal Opportunity, contudo, possui resultados insatisfatórios (abaixo do corte de 0.80) para todas as demais métricas utilizadas.

## RECOMENDAÇÕES

1. Este modelo possui uma variação significativa de performance entre subgrupos do atributo sensível (“Gênero"). De acordo com nossa análise quantitativa, métricas como Statistical Parity Ratio, Disparate Impact Score e Predictive Equality Ratio demonstram que quando pertencentes do subgrupo “Masculino" possuem uma maior chance de serem classificados com um baixo score de crédito (“Não Aprovado").
2. Caso este modelo seja utilizado, sem a devida supervisão humana, em aplicações que possam causar impacto a vida de indivíduos (e.g., concessão de benefícios), o modelo pode vir a discriminar contra indívíduos pertencentes do subgrupo "Gênero" : "Masculino". Existe a possibilidade de que as amostras pertencentes ao gênero "Masculino" do Credit Approval Data Set sejam mais propensas a receber uma classificação negativa, dado a maior presença de amostras pertencentes ao gênero "Masculino" rotuladas como "Não Aprovado" neste conjunto de dados (271: 112).
