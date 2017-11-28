# Simulated_Annealing
# Wat betekend het maken van een "fit"? 

We hebben een dataset $ ( s_{\rm{data}}, P_{\rm{clv}}^{\rm{data}})$ van sequenties ($s_{i}$) en bijbehorrende gemeten kansen ($P$). Nu willen we de parameter waardes van een model, in ons geval $P_{\rm{clv}}$, vinden dusdanig dat "de uitkomst van het model het meest lijkt op die van de gemeten data".

De maat die wij hiervoor gebruiken is de totale (kwadratische) afstand tussen de twee:


$$ 
\chi^{2} = \sum_{i} \frac{ (P_{\rm{clv}}^{\rm{data}} - P_{\rm{clv}}^{\rm{model}})^{2}  }{\sigma_{i}}
$$ 
waarbij $\sigma_{i}$ de onzekerheid is in meetwaarde $i$. Meetwaardes waarbij je een relatief hoge onzekerheid hebt tellen dus minder zwaar mee. 

Als we even de technische notatie laten voor wat het is hebben we in principe een functie $f(\vec{X})$ die afhangt van een set van parameters $\vec{X}$ die we willen minimaliseren (Nu heeft $\vec{X}$ dus componenten $p_{max}$, $n_{seed}$ en $\Delta_C$ voor jullie fit).Alle meetwaarden en onzekerheden zijn slechts constante parameters bekeken vannaf $f$). 

Vraag is nu dus, hoe gebruiken we de computer om het minimum te bepalen? 



# Simulated Annealing in een notendop 

Simulated annealing is een techniek die zich ontleent aan statistische physica. Een systeem met energieniveaus $V(\vec{X})$ heeft een kans om zich in equillibrium in toestand $\vec{X}$ te bevinden evenredig aan de bijbehorrende Boltzmann factor: 

$$ 
P(\vec{X}) \propto e^{-V(\vec{X}) / {k_BT}}
$$ 

Als je de omgeving langzaam genoeg laat afkoelen, dan zal het systeem zich uitleindelijk bevinden in de toestand met de laagste energie. De term "annealing" slaat op een techniek van de metaalindustrie: Men wist dat als je te snel het systeem afkoelt, dan wordt het metaal zeer breekbaar. Koel je langzaam, dan krijg je staal van goede kwaliteit. 

Wat als we nu even doen alsof onze functie $f(\vec{X})$ een energielandscap voorstelt. 

$$ 
f(\vec{X}) \leftrightarrow V(\vec{X})
$$ 

Als we dan artificieel een temperatuur parameter introduceren (vandaar "simmulated" in de naam)  kunnen we het minimum van de functie vinden door langzaam deze parameter in waarde af te laten nemen en te blijven zoeken naar de meest waarschijnlijke toestand volgens de Boltzmann verdeling. 

De code doet in principe het volgende:
1.  Start in een punt $\vec{X}$
2. Kies een nieuw "test punt" $\vec{X}'$
3. Accepteer $\vec{X}'$ als "nieuw startpunt" met een kans 
 $$ 
 p_{\rm{acc}} = \min[1, \frac{e^{-V(\vec{X}') / {k_BT}}}{ e^{-V(\vec{X}) / {k_BT}}  }    ]
 $$ 
Dus je accepteert altijd een toestand die meer waarschinlijk is, maar crucial is dat er een kans is om een minder waarschinlijke toestand ook te accepteren (dit zorgt ervoor dat je kunt onstnappen uit locale minima). 
4. Herhaal bovenstaande vaak terwijl je langzaam de temperatuur ($k_BT$) af laat nemen
5. Stop als je je ingestelde minimum Temperatuur hebt bereikt. 








