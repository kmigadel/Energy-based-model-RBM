# Modèles Génératifs d'images

## Modèles à énergie

#### Définition

Un modèle à énergie pour energy based-model (EBM) est une forme de modèle génératif. Le fonction des modèles génératifs sont simple, l'ocjectif est d'appredre une distribution de données en analysant un ensemble de données. Une fois la distribution apprise, le modèle est capable de générer de nouvelles données à l'aide de la distribution de données. Un exemple d'application des modèles à énergie est la génération de motifs particuliers (chiffre, lettre, logo, etc.) à l'aide d'images données en entrée.

#### Formules Générales

Le modèle à énergie porte ce nom grâce à la fonction qu'il implémente nommée fonction à énergie, notée $E_\theta(x)$ qui dispose des caractéristiques suivantes :

$$
E_{\theta}(v, h) = 0 \space (good) \\
E_{\theta}(v, h) > 0 \space (bad)
$$

Ainsi, la fonction à énergie renvoie un scalaire de probabilité non normalisé qu'on nomme énergie. Ce scalaire peut être proche ou éloigné de $0$ en fonction des paramètres du modèle $ \theta = (W, a, b)$. L'objectif est de trouver des paramètres qui minimisent l'énergie à l'aide d'un ensemble de données (variables visibles). 

Prenons $ E_{\theta}(x) $ une fonction d'énergie, ici, nous utiliserons la fonction Softmax pour normaliser les données entre $ [0,1] $ et permettre que leur somme soit égale à 1. Ainsi, nous pourrons définir $ \mathbb{P}_\theta(v,h) $ suivant une loi de probabilité à l'aide de la fonction suivante :

$$
\mathbb{P}_\theta(v, h) = \frac{e^{-E_{\theta}(v, h)}}{Z_\theta} = \frac{e^{-E_{\theta}(v, h)}}{\int_h \int_v e^{-E_{\theta}(v, h)}dv} \space , \space\space\space Z_\theta = \int_h \int_v e^{-E_{\theta}(v, h)}dv
$$

Sachant que $h$ prend ses valeurs dans l'ensemble des valeurs {$0,1$} nous nous retrouvons dans un cas discret, il nous est alors possible de transformer l'intégrale en somme.

$$
Z_\theta = \sum_{v, h} e^{-E_\theta(v, h)}
$$

En général, $ Z_\theta $ n'est pas calculable, car cette somme est simplement trop grande. On somme sur toutes les combinaisons de $v$ et $h$.

#### Entraînement du modèle

Pour entraîner un modèle à énergie, on a besoin d'un ensemble d'observations qu'on notera $\mathcal{D}$, composé d'un nombre $N$ d'observations, où les observations $ x_i $ sont indépendantes et identiquement distribuées (iid).

$$
\mathcal{D} = \{x_i\}_{i = 1}^{N}, \space\space\space i.i.d. \longrightarrow x_i ~ \mathcal{P}
$$

Dans un cas d'usage, notre ensemble d'observations $ \mathcal{D} $ peut être un ensemble d'images ($x_i$) représentant un digit (soit zéro, soit un, soit deux, etc.). Notre modèle cherchera alors les paramètres $\theta  = (W, a, b) $ optimaux qui permettront la génération de nouvelles observations qui ressemblent à celles fournies dans le jeu de donnée initial. Les paramètres $\theta$ optimaux sont ceux qui maximisent la loi de probabilité $\mathbb{P}_\theta(\mathcal{D})$. La représentation mathématique peut s'écrire de la manière suivante :

$$
\underset{\theta}{argmax} \space \mathbb{P}({\mathcal{D}|\theta}) = \underset{\theta}{argmax} \space \mathbb{P}_{\theta}(\mathcal{D}) = \underset{\theta}{argmax} \space \prod_{i=1}^N \mathbb{P}_{\theta}(x_i)
$$

Cependant dans la pratique, sommer autant de probabilités nous fait prendre le risque de faire tomber notre somme à zéro. Pour remédier à ce problème, il nous est possible d'utiliser le logarithme népérien, ici représenté par sa forme anglophone $\log$.

$$
\underset{\theta}{argmax} \space \log \prod_{i=1}^N \mathbb{P}_{\theta}(\mathcal{x_i}) = \underset{\theta}{argmax} \space \sum_{i=1}^N \log \mathbb{P}_{\theta}(x_i)
$$

Grâce à la propriété du $ log $, nous pouvons redéfinir $ \mathbb{P}_{\theta}(v) $ sous la forme suivante :

$$
\log \mathbb{P}_{\theta}(v) = \log(e^{-E_{\theta}(v)}) \log(\frac{1}{\int e^{-E_{\theta}(v)}dv}) \space , \space\space\space \log(\frac{1}{v}) = - \log(v)
$$

$$
\log \mathbb{P}_{\theta}(v) = -E_{\theta}(v) -\log(\int e^{-E_{\theta}(v)}dv)
$$


$$
\log \mathbb{P}_{\theta}(v) = -E_{\theta}(v) -\log(Z_\theta)
$$

Ne pouvant pas calculer $ \log(Z_\theta) $, nous devons recourir à un échantillonage afin d'estimer nos paramètres $\theta$. Notre modèle devra désormais appliquer une descente de gradient stochastique sur notre loi de probabilité par rapport à nos paramètres $\theta$. 

$$
\nabla_{\theta} \log \mathbb{P}({\mathcal{D}|\theta}) = \sum_{i=1}^N \nabla_{\theta} \log \mathbb{P}_{\theta}(v_i)
$$

Or $ \log \mathbb{P}_{\theta}(v) = -E_{\theta}(v) - \log(Z_\theta) $

$$
\nabla_{\theta} \log \mathbb{P}_{\theta}(v) = -\nabla_{\theta}E_{\theta}(v) -\frac{\nabla_{\theta}Z_\theta}{Z_\theta}, \space\space\space \nabla_{\theta} \log f_{\theta} = \frac{1}{f_{\theta}} \nabla_{\theta} f_{\theta}
$$

$$
\nabla_{\theta} \log \mathbb{P}_{\theta}(v) = -\nabla_{\theta}E_{\theta}(v) -\frac{\int \nabla_{\theta}e^{-E_{\theta}(v)}dv}{Z_\theta}
$$

$$
\nabla_{\theta} \log \mathbb{P}_{\theta}(v) = -\nabla_{\theta}E_{\theta}(v) +\int \nabla_{\theta}E_{\theta}(v) \frac{e^{-E_{\theta}(v)}}{Z_\theta}dv \space , \space\space\space \frac{e^{-E_{\theta}(v)}}{Z_\theta} = \mathbb{P}_{\theta}(v) 
$$

$$
\nabla_{\theta} \log \mathbb{P}_{\theta}(v) = -\nabla_{\theta}E_{\theta}(v) + \mathbb{E}_{\mathbb{P}_{\theta}}(\nabla_{\theta}E_{\theta}(c))
$$

Ainsi, on peut redéfinir $ \nabla_{\theta} \log \mathbb{P}({\mathcal{D}|\theta}) $ :

$$
\nabla_{\theta} \log \mathbb{P}({\mathcal{D}|\theta}) = \sum_{i=1}^N \nabla_{\theta} \log \mathbb{P}_{\theta}(v_i) = \sum_{i=1}^N -\nabla_{\theta}E_{\theta}(v_i) + N \mathbb{E}_{\mathbb{P}_{\theta}}(\nabla_{\theta}E_{\theta}(v))
$$

## Machine de Boltzmann Restreinte (RBM)

Pour mieux conceptualiser ce qui va suivre, nous allons nous placer dans un cas concrêt. Prenons un jeu de données composé d'images représentant des visages. Nous souhaiterions connaître la distribution de probabilités sous-jacentes à ces images, afin d'en tirer un modèle capable de générer de nouveaux visages. Ainsi, l'objectif n'est pas d'apprendre les observations par cœur, mais de généraliser ces observations pour en générer de nouvelles.

Une manière d'aborder le sujet serait d'utiliser les $\textbf{Machines de Boltzmann Restreintes}$, qui font partie des modèles qui implèmente les modèles à énergie. L'architecture d'un RBM peut être observée dans l'image ci-dessous.

![RBM](https://github.com/kmigadel/Energy-based-model-RBM/blob/master/assets/img/RBM.png)

Comme le démontre l'image ci-dessus, l'architecture est définie de la manière suivante :

- Nous avons deux couches de neurones, nous avons un couche de neurones visibles, notée $ v $, dont la dimension  $ d $ est équivalente à la taille de nos données, dans notre exemple ce sont le nombre de pixels dans une image. Chaque unité de la couche visible prend des valeurs dans $ \{0, 1\} $. Nous avons aussi une couche de neurones cachés, notée $ h $, , dont la dimension $ p $ est défini par l'utilisateur. Chaque unité de la couche cachée prend également ses valeurs dans $ \{0, 1\}$. Une unité cachée peut être conceptualisée comme la représentation d'une certaine caractéristique de nos données, et on les utilise pour pouvoir discerner ce qui caractéristique la distribution de probabilité de nos données que l'on cherche à modéliser. 

- L'architecture du réseau est un graphe biparti, c'est à dire que les neurones visibles sont connectés à tous les neurones cachés et inversement, mais il n'existe pas de liaisons entre elles. C'est à dire, les interactions visible-visible et caché-caché ne sont pas possibles. De ce fait, pour passer d'un état à un autre, on devra utiliser les probabilités conditionnelles $p(h|v) $ et $ p(v|h)$. On remarquera une ressemblance frappante avec entre la structure du RBM et celle d'un modèle de Markov cachée).

- Parmi ces deux couches nous avons une matrice de poids $ W $ qui définit la "force" des connexions entre les deux couches. Additionnellement aux poids, nous avons des paramètres de biais sur chaque couche de notre RBM, nous noterons $a$ le biais associé à la couche visible $v$ et $b$ le biais associé à la couche cachée $h$.

- Cas complet : Un RBM dans un cas complet dispose de la couche visible (observations) et des caractéristiques des observations (couche cachée). Ainsi, à la fin de son entrainement on connaitra l'influence des caractéristiques des observations (couche cachée) sur les observations.  

- Cas partiel : Un RBM dans un cas partiel ne dispose que de la couche visible (observations) et on laisse le RBM déterminer les caractéristiques des observations (couche cachée). Ainsi, après avoir été entrainé notre RBM nous saurons généré de nouvelles observations, mais nous ne saurons pas l'influence des caractéristiques des observations (couche cachée) sur les observations.

Dans le cas d'un RBM la fonction à énergie introduite précédemment et notée $E_\theta(v, h)$ est définie comme suit :

$$
\begin{equation}
E_\theta(v, h) = - \sum_i a_iv_i - \sum_j b_jh_j - \sum_{i,j} w_{i,j}v_ih_j \\
               = - a^Tv - b^Th - v^TWh
\end{equation}
$$

Ainsi, cette formule nous dit que lorsqu'un neurone de la couche visible $v$ prend la valeur $1$, il impactera le calcul de la fonction énergie, il en va de même pour les neurones de la couche cachée. On prend également en compte, à l'aide de $ \sum_{i,j} w_{i,j}v_ih_j $, l'ensemble des liaisons dont les neurones de la couche visible et cachée valent $1$. L'impact de cette somme dépendra du poid $W$ de cette liaison.

Ainsi nous pouvons definir les probabilités conditionnelles de la manière suivante :


$$
\mathbb{P}(h | v) = \frac{\mathbb{P}(h \cap v)}{\mathbb{P}(v)} = \frac{\frac{e^{-E_{\theta}(v,h)}}{Z_\theta}}{\frac{\displaystyle \sum_{h'\in\{0,1\}} e^{-E_{\theta}(v,h')}}{Z_\theta}} \space = \frac{e^{-E_{\theta}(v,h)}}{\displaystyle \sum_{h'\in\{0,1\}} e^{-E_{\theta}(v,h')}}
$$

$$
= \frac{e^{-(- a^Tv - b^Th - v^TWh)}}{\displaystyle \sum_{h' \in \{0,1\}} e^{-(- a^Tv - b^Th' - v^TWh)}}
$$

$$
= \frac{e^{a^Tv}e^{b^Th + v^TWh}}{e^{a^Tv} \displaystyle \sum_{h' \in \{0,1\}^p} e^{b^Th' + v^TWh'}}
$$

$$
= \frac{e^{a^Tv}e^{b^Th + v^TWh}}{e^{a^Tv} \displaystyle \sum_{h' \in \{0,1\}^p} e^{b^Th' + v^TWh'}}
$$

$$
= \frac{\displaystyle \prod_{j = 1}^p e^{h_jw_jv \space + \space b_jh_j}}{\displaystyle \sum_{h' \in \{0,1\}^p} \prod_{j = 1}^p e^{h'_jw_jv \space + \space b_jh'_j}}, \space\space\space v^TWh + b^Th = \sum_{j = 1}^p h_jw_jv +b_jh_j
$$

$$
= \frac{\displaystyle \prod_{j = 1}^p e^{h_jw_jv \space + \space b_jh_j}}{\displaystyle \prod_{j = 1}^p \sum_{h' \in \{0,1\}^p} e^{h'_jw_jv \space + \space b_jh'_j}}
$$

$$
= \prod_{j = 1}^p \space \frac{e^{h_jw_jv \space + \space b_jh_j}}{1 \space + \space e^{w_jv \space + \space b_j}} = \prod_{j = 1}^p \mathbb{P}(h_j | v)
$$
D'où :
$$
\mathbb{P}(h | v) = \prod_{j = 1}^p \mathbb{P}(h_j | v)
$$

Dans notre cas, on s'intèrese uniquement aux probabilités conditionnelles suivantes : $ \mathbb{P}(h_j = 1 | v) $ et $ \mathbb{P}(v_i = 1 | h) $. On peut donc déduire à l'aide du calcul fait précédemment que :

$$
\mathbb{P}(h_j = 1 | v) = \frac{1}{1 + e^{-w_jv - b_j}} = sigmoid(w_jv + b_j)
$$

De ce fait, on peut déduire qu'ayant $v$, on peut retirer des échantillons conditionnellement indépendants de $h$. De façon similaire, on peut déduire les expressions de $\mathbb{P}(v | h)$ et $\mathbb{P}(v_i = 1 | h)$ :

$$
\mathbb{P}(v | h) = \prod_{i = 1}^d \mathbb{P}(v_i | h)
$$

$$
\mathbb{P}(v_i = 1 | h) = sigmoid(w_ih + a_i)
$$

Ayant un $h$, il nous est donc possible d'échantilloner les valeurs conditionnelles des unités visibles. Maintenant que nous connaissons la loi de probabilité du couple et les probabilités conditionnelles, nous pouvons définir $\mathbb{P}_\theta(v)$.

Ceci signifie que si l'on prend la distribution de probabilité $\mathbb{P}_{\theta}(v,h)$ définie précédemment et que l'on marginalise les unités cachées (donc on intègre ou somme sur toutes les valeurs possibles des unités cachées), la distribution de probabilité marginale sur les unités visibles devrait correspondre à ce qui a été observé. Une fois cette étape franchie, on cherchera à échantilloner de notre modèle à énergie, en générant de nouveaux échantillons de $v$.


$$
\mathbb{P}_\theta(v) = \int_h \mathbb{P}_\theta(v,h)dh = \sum_{h'\in\{0,1\}} \mathbb{P}_\theta(v,h')
$$

$$
\mathbb{P}_\theta(v) = \frac{\sum_{h'\in\{0,1\}} e^{-E_{\theta}(v,h')}}{Z_\theta} \space , \space\space\space E_{\theta}(v,h) = - a^Tv - b^Th - v^TWh
$$

$$
\mathbb{P}_\theta(v) = \frac{\sum_{h'\in\{0,1\}} e^{-(-h'Wv^T - a^Tv - b^Th')}}{Z_\theta}
$$

$$
\mathbb{P}_\theta(v) = \frac{\sum_{h'\in\{0,1\}} e^{(h'Wv^T + a^Tv + b^Th')}}{Z_\theta}
$$

$$
\mathbb{P}_\theta(v) = \frac{e^{a^Tv} \sum_{h'\in\{0,1\}} e^{(h'Wv^T + b^Th')}}{Z_\theta}
$$

$$
\mathbb{P}_\theta(v) = \frac{e^{a^Tv} e^{\sum_{j=1}^p \log(1+e^{(W_j v + b_j)})}}{Z_\theta}
$$

$\mathbb{P}_\theta(v)$ est la distribution marginale des unités visibles. Ceci signifie que l'on prend la distribution de probabilité $\mathbb{P}_\theta(v, h)$ et l'on marginalise les unités cachées. Évaluer cette distribution de probabilité n'est pas simple, on ne peut donc pas écrire explicitement la forme de $\mathbb{P}_\theta(v)$ mais il nous est possible d'écrire formellement la forme fonctionnelle de $E_\theta(v)$ (que l'on peut déduire à l'aide du calcul effectué ci-dessus).

$$
E_{\theta}(v) = -(a^Tv + \sum_{j=1}^p \log(1+e^{(W_j v + b_j)}))
$$ 

$E_{\theta}(v)$ contient tous les ordres d'interactions entre les unités visibles. Ceci signifie que l'on peut représenter arbitrairement des distributions de probabilité complexes dans $v$.
Si l'on met le nombre d'unités cachées arbitrairement grand, on peut en principe approximer n'importe quelle distribution de probabilité des unités visibles avec une précision arbitrairement bonne, mais ceci ne signifie malheureusement pas qu'il sera facile de retrouver les paramètres qui réaliserons cette bonne approximation.

#### Entraînement du modèle 

En calculant la fonction d'énergie, nous nous intéresserons au minimum local de cette dernière. Ainsi, lors de la phase d'entraînement du modèle, nous allons faire observer une partie de notre jeu de données (ici, des images de digit), et faire varier les valeurs de nos paramètres $\theta = (a, b, W)$ afin d'approcher un mimimum local.

![Energy](https://github.com/kmigadel/Energy-based-model-RBM/blob/master/assets/img/Energy.png)

On commencera de ce fait par un vecteur de features (par exemple une image d'un digit) et on minimisera l'énergie, en trouvant une nouvelle paire de $v$ et $h$ où nos vecteurs visibles auront été modifiés dans la figure ci-dessus on dit qu'ils se sont rapprochés du minimum d'énergie, ici représenté par les points rouges sur le graphe.

  

Pour pouvoir minimiser la fonction à énergie, nous pouvons nous appuyer sur l'expression du gradient stochastisque définie dans les modèles à énergie.

$$
\nabla_{\theta} \log \mathbb{P}({\mathcal{D}|\theta}) = \sum_{i=1}^N \nabla_{\theta} \log \mathbb{P}_{\theta}(v_i) = \sum_{i=1}^N -\nabla_{\theta}E_{\theta}(v_i) + N \mathbb{E}_{\mathbb{P}_{\theta}}(\nabla_{\theta}E_{\theta}(v))
$$

Cenpendant, le terme $ \mathbb{E}_{\mathbb{P}_{\theta}}(\nabla_{\theta}E_{\theta}(v)) $ ne peut pas être calculé, à cause de sa dépendance avec $Z_\theta$. Il nous est cependant possible de l'approximer à l'aide de la méthode du MCMC (Markov Chain Monté Carlo) donnée ci-dessous :

$$
\underset{X \backsim q}{\mathbb{E}}f(x) = \int_{x \in \Omega} q(x)f(x)dx \approx \frac{1}{N} \underset{x_i \backsim q}{\sum_{i=1}^N f(x_i)}
$$

Ainsi, nous pouvons réécrire la formule $ \mathbb{E}_{\mathbb{P}_{\theta}}(\nabla_{\theta}E_{\theta}(v)) $ de la manière suivante :

$$
\mathbb{E}_{v \backsim \mathbb{P}_{\theta}}(\nabla_{\theta}E_{\theta}(v)) \approx \frac{1}{M} \underset{v_i \backsim \mathbb{P}_{\theta}}{\sum_{i=1}^M \nabla_{\theta}E_{\theta}(v_i)}
$$

Nous pouvons à présent calculer $ \nabla_{\theta} \log\mathbb{P}({\mathcal{D}|\theta})) $ :

$$
\nabla_{\theta} \log\mathbb{P}({\mathcal{D}|\theta})) = - \underset{v_i \backsim \mathcal{D}}{\sum_{i=1}^N \nabla_{\theta}E_{\theta}(v_i)} + \frac{N}{M} \underset{\tilde{v_i} \backsim \mathbb{P}_{\theta}}{\sum_{i=1}^M \nabla_{\theta}E_{\theta}(\tilde{v_i})}
$$

$$
\nabla_{\theta} \log\mathbb{P}({\mathcal{D}|\theta})) = \nabla_{\theta} (- \underset{v_i \backsim \mathcal{D}}{\sum_{i=1}^N E_{\theta}(v_i)} + \frac{N}{M} \underset{\tilde{v_i} \backsim \mathbb{P}_{\theta}}{\sum_{i=1}^M E_{\theta}(\tilde{v_i})})
$$



Ainsi, on peut définir notre fonction de minimisation de $ - \log \mathbb{P}_{\theta}(\mathcal{D}) $ comme :

$$
\underset{\theta}{argmin} \space \nabla_{\theta} - \log \mathbb{P}_{\theta}(\mathcal{D}) = \nabla_{\theta} (\underset{x_i \backsim \mathcal{D}}{\sum_{i=1}^N E_{\theta}(v_i)} - \frac{N}{M} \underset{\tilde{v_i} \backsim \mathbb{P}_{\theta}}{\sum_{i=1}^M E_{\theta}(\tilde{v_i})})
$$


On définit notre fonction de perte, notée $ \mathcal{L}(\theta) $ de la manière suivante : 
$$
\mathcal{L}(\theta) = \underset{v_i \backsim \mathcal{D}}{\sum_{i=1}^N E_{\theta}(v_i)} - \frac{N}{M} \underset{\tilde{v_i} \backsim \mathbb{P}_{\theta}}{\sum_{i=1}^M E_{\theta}(\tilde{v_i})}
$$

Référence :

Energy-based models : https://www.youtube.com/watch?v=kpulMklVmRU