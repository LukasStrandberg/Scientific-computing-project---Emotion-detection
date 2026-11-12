# Projektlogg

## 2026/11/03
Idag skapade vi ett projekt på github, LaTeX-rapport på overleaf och började sätta oss in i literaturen genom att läsa rapporten av Minyu Chen. Vi hade även uppstartsmöte med Stefan och Anders. Började även kolla på på mediapipe och simpla modeller på annoterade dataset från internet

## 2026/11/04
Kollade mer på andras verk, hittade denna artikel "From static to dynamic: Adapting landmark-aware image models for facial expression recognition in videos" som fått bra resultat och kombinerar ViT och landmarks.

## 2026/11/09
Finjusterade MAE-DFER's förtränade model på på datasetet ElderReact i hopp om att få bättre resultat på äldre människor.

## 2026/11/10
Möte med Anders för statusuppdatering. Diskuterade tillvägagångssätt och resultat än så länge: Clustering, MAE-DFER. Även tillgång till beräkningsresurser och mer data.

## 2026/11/11
Jobbade på implementering av s2d pipeline från "From static to dynamic: Adapting landmark-aware image models for facial expression recognition in videos". Smoketest visar något sämre performance än artikeln men bättre än Minyu Chen. Kan vara några buggar att reda ut. Nästa steg är att fine-tunea med ElderReact och sedan testa på vår data.

## 2026/11/12
Lyckades få predictions på vår riktiga testdata från MAE-DFER finjusterade modell från ELderReact data samt den färdigtränade MAE-DFER. Resultat är inte superbra, stor majoritet av känslorna den fångar är "ledsen", vilket är svårt då hon har vissa ledsna tendenser utan att hon nödvändigtvis är ledsen. Vi kan även se vad modellen kollar på och det är inte alltid ansiktet, detta måste undersökas mera. Funderar även på att ta bort ett slags "medelvärde" av ansiktet, för att motverka hennes ledsna tendenser.
