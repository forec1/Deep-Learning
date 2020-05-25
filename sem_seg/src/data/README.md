Kako radim s jako malim skupom za treniranje, generirao sam još slika primjenom
nekoliko tehnika transformacije na postojeće slike koje pripadaju skupu za treniranje:

– slike su izrezane na slučajno odabranim mjestima

– s 50% šanse su vertikalno i horizontalno preokrenute

– svjetlina je promjenjena za ∆ iz [−0.2, 0.2]

Postupak sam za svaku sliku ponovio 9 puta nakon čega sam dobio 4113 novih slika. 
Uz to, kako bi mogao koristiti grupno učenje (engl.  batch), morao sam popuniti sve
slike do HxW gdje su H i W maksimalna visina, odnosno širina (320, 320).