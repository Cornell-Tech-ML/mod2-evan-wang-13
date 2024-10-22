[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py



### Dataset: Simple
Size of Hidden Layer: 2

0.5 Learning Rate, 500 Epochs

Time per epoch: 0.070s

50 datapoints


<details>
<summary>Training Logs Dropdown </summary>

```bash
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 32.75813070109494, correct: 32
Epoch: 20/500, loss: 24.43077012652386, correct: 43
Epoch: 30/500, loss: 15.028873750831334, correct: 47
Epoch: 40/500, loss: 12.466059482976185, correct: 46
Epoch: 50/500, loss: 17.16238383447687, correct: 42
Epoch: 60/500, loss: 14.516494562152573, correct: 42
Epoch: 70/500, loss: 14.489091507252647, correct: 43
Epoch: 80/500, loss: 18.374102129096244, correct: 42
Epoch: 90/500, loss: 22.04533140872072, correct: 40
Epoch: 0/500, loss: 0, correct: 0
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 33.085072654200104, correct: 33
Epoch: 20/500, loss: 32.01533650836951, correct: 37
Epoch: 30/500, loss: 31.262500037666353, correct: 38
Epoch: 40/500, loss: 30.299791978832587, correct: 38
Epoch: 50/500, loss: 29.29767800834028, correct: 38
Epoch: 60/500, loss: 28.238110934369907, correct: 34
Epoch: 70/500, loss: 29.14354156688937, correct: 29
Epoch: 80/500, loss: 28.565221464765443, correct: 29
Epoch: 90/500, loss: 25.901071702098694, correct: 34
Epoch: 100/500, loss: 22.098597647131186, correct: 41
Epoch: 110/500, loss: 20.596267935313648, correct: 41
Epoch: 120/500, loss: 13.483357844968689, correct: 46
Epoch: 130/500, loss: 10.25931306936417, correct: 47
Epoch: 140/500, loss: 7.861913428776001, correct: 50
Epoch: 150/500, loss: 6.5164267868937875, correct: 50
Epoch: 160/500, loss: 5.527057718680876, correct: 50
Epoch: 170/500, loss: 4.779656719908626, correct: 50
Epoch: 180/500, loss: 4.292175259605807, correct: 50
Epoch: 190/500, loss: 3.978260229914049, correct: 50
Epoch: 200/500, loss: 3.753066098623981, correct: 50
Epoch: 210/500, loss: 3.5396441299361547, correct: 50
Epoch: 220/500, loss: 3.3283067668652735, correct: 50
Epoch: 230/500, loss: 3.1191846720188185, correct: 50
Epoch: 240/500, loss: 2.915754512350239, correct: 50
Epoch: 250/500, loss: 2.7241613244362246, correct: 50
Epoch: 260/500, loss: 2.548095243393215, correct: 50
Epoch: 270/500, loss: 2.3887569723805067, correct: 50
Epoch: 280/500, loss: 2.245615615066134, correct: 50
Epoch: 290/500, loss: 2.11722486995449, correct: 50
Epoch: 300/500, loss: 2.0018143978661262, correct: 50
Epoch: 310/500, loss: 1.9009867801415683, correct: 50
Epoch: 320/500, loss: 1.7980849660560523, correct: 50
Epoch: 330/500, loss: 1.710365839166617, correct: 50
Epoch: 340/500, loss: 1.6309519743862047, correct: 50
Epoch: 350/500, loss: 1.558456253194815, correct: 50
Epoch: 360/500, loss: 1.4918284008391385, correct: 50
Epoch: 370/500, loss: 1.4302219269005148, correct: 50
Epoch: 380/500, loss: 1.3729596094015366, correct: 50
Epoch: 390/500, loss: 1.3195017207397683, correct: 50
Epoch: 400/500, loss: 1.2694170116993273, correct: 50
Epoch: 410/500, loss: 1.2223582389440806, correct: 50
Epoch: 420/500, loss: 1.1780421768275595, correct: 50
Epoch: 430/500, loss: 1.1362338138780692, correct: 50
Epoch: 440/500, loss: 1.0967342420794803, correct: 50
Epoch: 450/500, loss: 1.0593716488841514, correct: 50
Epoch: 460/500, loss: 1.0239947943965162, correct: 50
Epoch: 470/500, loss: 0.9904683904374803, correct: 50
Epoch: 480/500, loss: 0.9586698750499585, correct: 50
Epoch: 490/500, loss: 0.9284871725386169, correct: 50
Epoch: 500/500, loss: 0.8998171263704744, correct: 50
```

</details>

### Dataset: Diag
Size of Hidden Layer: 2

1.0 Learning Rate. 500 Epochs
Time per epoch: 0.070s.

50 datapoints


<details>
<summary>Training Logs Dropdown </summary>

```bash
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 18.562579861019504, correct: 44
Epoch: 20/500, loss: 17.625560071951362, correct: 44
Epoch: 30/500, loss: 15.608805702283806, correct: 44
Epoch: 40/500, loss: 12.270587931973072, correct: 44
Epoch: 50/500, loss: 9.96032903943302, correct: 44
Epoch: 60/500, loss: 8.546390667266339, correct: 46
Epoch: 70/500, loss: 6.273082331873233, correct: 46
Epoch: 80/500, loss: 4.506870180914331, correct: 50
Epoch: 90/500, loss: 3.1794818148392126, correct: 50
Epoch: 100/500, loss: 2.3683283764186993, correct: 50
Epoch: 110/500, loss: 1.8679999668911336, correct: 50
Epoch: 120/500, loss: 1.5791379025542736, correct: 50
Epoch: 130/500, loss: 1.3232294047355684, correct: 50
Epoch: 140/500, loss: 1.1530863952648263, correct: 50
Epoch: 150/500, loss: 1.0211181407827432, correct: 50
Epoch: 160/500, loss: 0.9157471745303051, correct: 50
Epoch: 170/500, loss: 0.8305678108183747, correct: 50
Epoch: 180/500, loss: 0.7569203194646752, correct: 50
Epoch: 190/500, loss: 0.6972909759928617, correct: 50
Epoch: 200/500, loss: 0.643969265155579, correct: 50
Epoch: 210/500, loss: 0.5983876795250918, correct: 50
Epoch: 220/500, loss: 0.5587239653277615, correct: 50
Epoch: 230/500, loss: 0.5240028896077165, correct: 50
Epoch: 240/500, loss: 0.4948396363331007, correct: 50
Epoch: 250/500, loss: 0.46516345193088715, correct: 50
Epoch: 260/500, loss: 0.4393919212503357, correct: 50
Epoch: 270/500, loss: 0.4189161102085133, correct: 50
Epoch: 280/500, loss: 0.3981592917473147, correct: 50
Epoch: 290/500, loss: 0.3794025215472392, correct: 50
Epoch: 300/500, loss: 0.36006983177258095, correct: 50
Epoch: 310/500, loss: 0.3450028819874986, correct: 50
Epoch: 320/500, loss: 0.3294892381148225, correct: 50
Epoch: 330/500, loss: 0.3168808505610876, correct: 50
Epoch: 340/500, loss: 0.30369497991705585, correct: 50
Epoch: 350/500, loss: 0.2934060372962692, correct: 50
Epoch: 360/500, loss: 0.2819990202313904, correct: 50
Epoch: 370/500, loss: 0.27164718757533923, correct: 50
Epoch: 380/500, loss: 0.2620672000385223, correct: 50
Epoch: 390/500, loss: 0.2532184384595418, correct: 50
Epoch: 400/500, loss: 0.244952143253352, correct: 50
Epoch: 410/500, loss: 0.23727531341411795, correct: 50
Epoch: 420/500, loss: 0.23018236357861532, correct: 50
Epoch: 430/500, loss: 0.22233233988062018, correct: 50
Epoch: 440/500, loss: 0.21598462564975468, correct: 50
Epoch: 450/500, loss: 0.2100994483598697, correct: 50
Epoch: 460/500, loss: 0.20369426855574546, correct: 50
Epoch: 470/500, loss: 0.19835205160132022, correct: 50
Epoch: 480/500, loss: 0.19351017993753264, correct: 50
Epoch: 490/500, loss: 0.18794397486086498, correct: 50
Epoch: 500/500, loss: 0.1836431594994341, correct: 50
```
</details>

### Dataset: Split
Size of hidden layer: 4

0.5 Learning Rate, 500 Epochs

50 datapoints

![newplot (4)](https://github.com/user-attachments/assets/d28fdd8c-97f9-4a7d-bee5-e77ce2d71bfc)

![newplot (5)](https://github.com/user-attachments/assets/0fe5e782-d78e-4fd8-9815-642cd53c44f0)

<details>
<summary>Training Logs Dropdown</summary>

```bash
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 34.15689500598109, correct: 29
Epoch: 20/500, loss: 33.66285017848301, correct: 30
Epoch: 30/500, loss: 33.36149709314213, correct: 29
Epoch: 40/500, loss: 32.96663460320763, correct: 32
Epoch: 50/500, loss: 32.0701353459562, correct: 33
Epoch: 60/500, loss: 30.18790652561743, correct: 36
Epoch: 70/500, loss: 28.357384067580053, correct: 39
Epoch: 80/500, loss: 27.311992022915494, correct: 35
Epoch: 90/500, loss: 25.815495785398415, correct: 35
Epoch: 100/500, loss: 23.697352071722754, correct: 36
Epoch: 110/500, loss: 23.44439078354259, correct: 36
Epoch: 120/500, loss: 22.911285880139225, correct: 36
Epoch: 130/500, loss: 22.139324382472722, correct: 37
Epoch: 140/500, loss: 20.900471667777303, correct: 39
Epoch: 150/500, loss: 19.423989783225885, correct: 42
Epoch: 160/500, loss: 17.617791924232854, correct: 44
Epoch: 170/500, loss: 15.94960155559194, correct: 45
Epoch: 180/500, loss: 12.339323517314249, correct: 46
Epoch: 190/500, loss: 11.939565755686429, correct: 46
Epoch: 200/500, loss: 13.498101691652296, correct: 45
Epoch: 210/500, loss: 7.217603462353987, correct: 48
Epoch: 220/500, loss: 4.62559971980771, correct: 50
Epoch: 230/500, loss: 4.108551276448302, correct: 50
Epoch: 240/500, loss: 3.69660592615835, correct: 50
Epoch: 250/500, loss: 3.365022547104043, correct: 50
Epoch: 260/500, loss: 3.0690457793177424, correct: 50
Epoch: 270/500, loss: 2.8150514920194203, correct: 50
Epoch: 280/500, loss: 2.5907017701523825, correct: 50
Epoch: 290/500, loss: 2.3914623366212244, correct: 50
Epoch: 300/500, loss: 2.213546608706639, correct: 50
Epoch: 310/500, loss: 2.054262105852168, correct: 50
Epoch: 320/500, loss: 1.9111543999941865, correct: 50
Epoch: 330/500, loss: 1.7821678724100825, correct: 50
Epoch: 340/500, loss: 1.6655659337822553, correct: 50
Epoch: 350/500, loss: 1.5606601280080448, correct: 50
Epoch: 360/500, loss: 1.4664626488508654, correct: 50
Epoch: 370/500, loss: 1.3857617104536684, correct: 50
Epoch: 380/500, loss: 1.3146601955615371, correct: 50
Epoch: 390/500, loss: 1.250634927106931, correct: 50
Epoch: 400/500, loss: 1.1908488858792854, correct: 50
Epoch: 410/500, loss: 1.131563531007657, correct: 50
Epoch: 420/500, loss: 1.0694124011998956, correct: 50
Epoch: 430/500, loss: 1.0081401745882943, correct: 50
Epoch: 440/500, loss: 0.9471906475241946, correct: 50
Epoch: 450/500, loss: 0.896227916385996, correct: 50
Epoch: 460/500, loss: 0.8532652629702373, correct: 50
Epoch: 470/500, loss: 0.8147181813049468, correct: 50
Epoch: 480/500, loss: 0.779017850536186, correct: 50
Epoch: 490/500, loss: 0.745821606982349, correct: 50
Epoch: 500/500, loss: 0.7147988514648224, correct: 50
```
</details>

### Dataset: Xor
Size of hidden layer: 8

0.1 Learning Rate, 900 Epochs

50 datapoints

![newplot (6)](https://github.com/user-attachments/assets/320ee475-f419-4866-83cd-9561dcf60b0e)

![newplot (7)](https://github.com/user-attachments/assets/bbbe8949-1158-4135-86ba-e1aff5b1e2e4)

<details>
<summary>Training Logs Dropdown </summary>

```bash
Epoch: 0/900, loss: 0, correct: 0
Epoch: 10/900, loss: 33.18489496695082, correct: 37
Epoch: 20/900, loss: 32.47750068749632, correct: 38
Epoch: 30/900, loss: 31.63200111749013, correct: 38
Epoch: 40/900, loss: 30.528668059547872, correct: 37
Epoch: 50/900, loss: 28.947619427306, correct: 38
Epoch: 60/900, loss: 27.128064263353714, correct: 37
Epoch: 70/900, loss: 25.0021311213535, correct: 38
Epoch: 80/900, loss: 23.264051334628313, correct: 39
Epoch: 90/900, loss: 22.020235534609245, correct: 40
Epoch: 100/900, loss: 21.063116596372417, correct: 40
Epoch: 110/900, loss: 20.246958454301943, correct: 41
Epoch: 120/900, loss: 19.5578185519227, correct: 42
Epoch: 130/900, loss: 18.90963606782121, correct: 41
Epoch: 140/900, loss: 18.33600402616595, correct: 42
Epoch: 150/900, loss: 17.817548100340606, correct: 42
Epoch: 160/900, loss: 17.384803024463675, correct: 42
Epoch: 170/900, loss: 16.998722110664417, correct: 42
Epoch: 180/900, loss: 16.652340867583757, correct: 42
Epoch: 190/900, loss: 16.322682750132085, correct: 42
Epoch: 200/900, loss: 16.00679339364102, correct: 42
Epoch: 210/900, loss: 15.70044642639935, correct: 42
Epoch: 220/900, loss: 15.30776005479259, correct: 43
Epoch: 230/900, loss: 14.99745672718418, correct: 43
Epoch: 240/900, loss: 14.696335291138123, correct: 43
Epoch: 250/900, loss: 14.410889344246037, correct: 43
Epoch: 260/900, loss: 14.134368931646083, correct: 43
Epoch: 270/900, loss: 13.869383178396028, correct: 43
Epoch: 280/900, loss: 13.606604758067819, correct: 43
Epoch: 290/900, loss: 13.348759682018855, correct: 43
Epoch: 300/900, loss: 13.090514091355626, correct: 43
Epoch: 310/900, loss: 12.833818305888325, correct: 43
Epoch: 320/900, loss: 12.5754607347958, correct: 43
Epoch: 330/900, loss: 12.316374066817053, correct: 43
Epoch: 340/900, loss: 12.056315205314776, correct: 44
Epoch: 350/900, loss: 11.799627937120718, correct: 44
Epoch: 360/900, loss: 11.550843389005253, correct: 44
Epoch: 370/900, loss: 11.316643218278331, correct: 46
Epoch: 380/900, loss: 11.06968876531855, correct: 46
Epoch: 390/900, loss: 10.8392331955595, correct: 46
Epoch: 400/900, loss: 10.612264037100521, correct: 46
Epoch: 410/900, loss: 10.387097872991784, correct: 47
Epoch: 420/900, loss: 10.16652303786754, correct: 47
Epoch: 430/900, loss: 9.952121892913372, correct: 47
Epoch: 440/900, loss: 9.74159618961797, correct: 47
Epoch: 450/900, loss: 9.538683756533006, correct: 47
Epoch: 460/900, loss: 9.341823065558833, correct: 47
Epoch: 470/900, loss: 9.14737452920977, correct: 47
Epoch: 480/900, loss: 8.957686154708972, correct: 47
Epoch: 490/900, loss: 8.779266515267464, correct: 47
Epoch: 500/900, loss: 8.591527648505561, correct: 47
Epoch: 510/900, loss: 8.409112805928553, correct: 47
Epoch: 520/900, loss: 8.228684869196366, correct: 48
Epoch: 530/900, loss: 8.057061767529904, correct: 48
Epoch: 540/900, loss: 7.8775008505148145, correct: 48
Epoch: 550/900, loss: 7.704631962861884, correct: 48
Epoch: 560/900, loss: 7.53685432695245, correct: 48
Epoch: 570/900, loss: 7.368977967093104, correct: 48
Epoch: 580/900, loss: 7.204090958526137, correct: 48
Epoch: 590/900, loss: 7.041004923884497, correct: 48
Epoch: 600/900, loss: 6.879941204620596, correct: 48
Epoch: 610/900, loss: 6.721651440224754, correct: 48
Epoch: 620/900, loss: 6.5655353599371535, correct: 48
Epoch: 630/900, loss: 6.41254242640532, correct: 48
Epoch: 640/900, loss: 6.263468386083932, correct: 48
Epoch: 650/900, loss: 6.115696697384342, correct: 50
Epoch: 660/900, loss: 5.973399789413323, correct: 50
Epoch: 670/900, loss: 5.831683778503563, correct: 50
Epoch: 680/900, loss: 5.695642652581029, correct: 50
Epoch: 690/900, loss: 5.5604163181779995, correct: 50
Epoch: 700/900, loss: 5.430732103079995, correct: 50
Epoch: 710/900, loss: 5.3054392965919535, correct: 50
Epoch: 720/900, loss: 5.182667900926031, correct: 50
Epoch: 730/900, loss: 5.060563079420299, correct: 50
Epoch: 740/900, loss: 4.943590319295232, correct: 50
Epoch: 750/900, loss: 4.831220682185909, correct: 50
Epoch: 760/900, loss: 4.720382781448805, correct: 50
Epoch: 770/900, loss: 4.612268084719478, correct: 50
Epoch: 780/900, loss: 4.509457714072454, correct: 50
Epoch: 790/900, loss: 4.404953992278249, correct: 50
Epoch: 800/900, loss: 4.303550586805878, correct: 50
Epoch: 810/900, loss: 4.205648681513955, correct: 50
Epoch: 820/900, loss: 4.112694967147237, correct: 50
Epoch: 830/900, loss: 4.01951288166861, correct: 50
Epoch: 840/900, loss: 3.929599411611363, correct: 50
Epoch: 850/900, loss: 3.8435399790942197, correct: 50
Epoch: 860/900, loss: 3.758510973176314, correct: 50
Epoch: 870/900, loss: 3.6796889597223204, correct: 50
Epoch: 880/900, loss: 3.59878831519473, correct: 50
Epoch: 890/900, loss: 3.52124885456271, correct: 50
Epoch: 900/900, loss: 3.4477596932205974, correct: 50

```

</details>