# Projektlogg

## 2026/11/03
Idag skapade vi ett projekt på github, LaTeX-rapport på overleaf och började sätta oss in i literaturen genom att läsa rapporten av Minyu Chen. Vi hade även uppstartsmöte med Stefan och Anders. Började även kolla på på mediapipe och simpla modeller på annoterade dataset från internet

## 2026/11/04
Kollade mer på andras verk, hittade denna artikel "From static to dynamic: Adapting landmark-aware image models for facial expression recognition in videos" som fått bra resultat och kombinerar ViT och landmarks.

## 2026/11/10
Möte med Anders för statusuppdatering. Diskuterade tillvägagångssätt och resultat än så länge: Clustering, MAE-DFER. Även tillgång till beräkningsresurser och mer data.

## 2026/11/11
Jobbade på implementering av s2d pipeline från "From static to dynamic: Adapting landmark-aware image models for facial expression recognition in videos". Smoketest visar något sämre performance än artikeln men bättre än Minyu Chen. Kan vara några buggar att reda ut. Nästa steg är att fine-tunea med ElderReact och sedan testa på vår data.
