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

0.1 Learning Rate, 500 Epochs

Time per epoch: 0.062s

50 datapoints

![newplot (19)](https://github.com/user-attachments/assets/640cb227-f951-4c36-a89d-b93554232d56)

![newplot (20)](https://github.com/user-attachments/assets/ac192779-bfdb-40a7-a565-1be016892950)


<details>
<summary>Training Logs Dropdown </summary>

```bash
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 34.70898923085915, correct: 27
Epoch: 20/500, loss: 34.598607383516075, correct: 27
Epoch: 30/500, loss: 34.490279602919664, correct: 27
Epoch: 40/500, loss: 34.38236593444117, correct: 27
Epoch: 50/500, loss: 34.27325976260079, correct: 27
Epoch: 60/500, loss: 34.16135073853456, correct: 29
Epoch: 70/500, loss: 34.04498981510594, correct: 32
Epoch: 80/500, loss: 33.92245482954228, correct: 33
Epoch: 90/500, loss: 33.79191531299636, correct: 35
Epoch: 100/500, loss: 33.651395343996526, correct: 35
Epoch: 110/500, loss: 33.49873334107697, correct: 34
Epoch: 120/500, loss: 33.33153774421369, correct: 35
Epoch: 130/500, loss: 33.15000599892139, correct: 37
Epoch: 140/500, loss: 32.95206706485186, correct: 39
Epoch: 150/500, loss: 32.733631672400044, correct: 41
Epoch: 160/500, loss: 32.491458879981664, correct: 41
Epoch: 170/500, loss: 32.222087338289626, correct: 41
Epoch: 180/500, loss: 31.92143221545088, correct: 42
Epoch: 190/500, loss: 31.58489760862497, correct: 42
Epoch: 200/500, loss: 31.20955922246449, correct: 42
Epoch: 210/500, loss: 30.792480072007347, correct: 43
Epoch: 220/500, loss: 30.3266789659021, correct: 43
Epoch: 230/500, loss: 29.80568396970564, correct: 43
Epoch: 240/500, loss: 29.23118851558359, correct: 43
Epoch: 250/500, loss: 28.60641597150595, correct: 44
Epoch: 260/500, loss: 27.91583333357125, correct: 44
Epoch: 270/500, loss: 27.165941625606777, correct: 45
Epoch: 280/500, loss: 26.35381696477112, correct: 45
Epoch: 290/500, loss: 25.487295063711258, correct: 45
Epoch: 300/500, loss: 24.599457099015844, correct: 46
Epoch: 310/500, loss: 23.698014383612193, correct: 46
Epoch: 320/500, loss: 22.802727814102674, correct: 46
Epoch: 330/500, loss: 21.924768785019033, correct: 45
Epoch: 340/500, loss: 21.093909422910585, correct: 45
Epoch: 350/500, loss: 20.337866552880207, correct: 45
Epoch: 360/500, loss: 19.612095434067918, correct: 45
Epoch: 370/500, loss: 18.910602851972126, correct: 45
Epoch: 380/500, loss: 18.24460154315586, correct: 46
Epoch: 390/500, loss: 17.602114149305066, correct: 46
Epoch: 400/500, loss: 17.017995680925782, correct: 46
Epoch: 410/500, loss: 16.47637653640042, correct: 47
Epoch: 420/500, loss: 15.957953867240924, correct: 47
Epoch: 430/500, loss: 15.461315732222971, correct: 47
Epoch: 440/500, loss: 14.995923469570188, correct: 47
Epoch: 450/500, loss: 14.560493302897255, correct: 47
Epoch: 460/500, loss: 14.154722643575592, correct: 47
Epoch: 470/500, loss: 13.76751752857026, correct: 47
Epoch: 480/500, loss: 13.397613571892345, correct: 48
Epoch: 490/500, loss: 13.043597607865435, correct: 50
Epoch: 500/500, loss: 12.705469493384516, correct: 50
```

</details>

### Dataset: Diag
Size of Hidden Layer: 2

0.5 Learning Rate. 500 Epochs

Time per epoch: 0.063s

50 datapoints

![newplot (21)](https://github.com/user-attachments/assets/ed4e1f7f-8641-476f-836b-a42eb81c6b42)

![newplot (22)](https://github.com/user-attachments/assets/635278b3-a17b-4e01-91e5-1bc52537a8ba)

<details>
<summary>Training Logs Dropdown </summary>

```bash
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 21.457335315369136, correct: 43
Epoch: 20/500, loss: 20.46870947126386, correct: 43
Epoch: 30/500, loss: 20.29097619201011, correct: 43
Epoch: 40/500, loss: 20.251489597439488, correct: 43
Epoch: 50/500, loss: 20.241313501260553, correct: 43
Epoch: 60/500, loss: 20.233253620117036, correct: 43
Epoch: 70/500, loss: 20.215522912358693, correct: 43
Epoch: 80/500, loss: 20.153329466217627, correct: 43
Epoch: 90/500, loss: 19.997719639962195, correct: 43
Epoch: 100/500, loss: 19.689322428729923, correct: 43
Epoch: 110/500, loss: 19.209163828858827, correct: 43
Epoch: 120/500, loss: 18.532197853864947, correct: 43
Epoch: 130/500, loss: 17.60983064451771, correct: 43
Epoch: 140/500, loss: 16.433334217223194, correct: 43
Epoch: 150/500, loss: 15.16335989892979, correct: 43
Epoch: 160/500, loss: 13.709130380431612, correct: 43
Epoch: 170/500, loss: 12.067346021895515, correct: 43
Epoch: 180/500, loss: 10.760031692814167, correct: 43
Epoch: 190/500, loss: 9.408732959027393, correct: 43
Epoch: 200/500, loss: 8.252885357115634, correct: 43
Epoch: 210/500, loss: 7.2274221873932225, correct: 43
Epoch: 220/500, loss: 6.3548986571638375, correct: 48
Epoch: 230/500, loss: 5.604856227802335, correct: 48
Epoch: 240/500, loss: 4.9593793039812955, correct: 49
Epoch: 250/500, loss: 4.395495421272724, correct: 49
Epoch: 260/500, loss: 3.9532083416662966, correct: 49
Epoch: 270/500, loss: 3.5466776136911635, correct: 50
Epoch: 280/500, loss: 3.233756115719806, correct: 50
Epoch: 290/500, loss: 3.001862980459684, correct: 50
Epoch: 300/500, loss: 2.736304640955198, correct: 50
Epoch: 310/500, loss: 2.537466049481232, correct: 50
Epoch: 320/500, loss: 2.360155926646505, correct: 50
Epoch: 330/500, loss: 2.206278814148092, correct: 50
Epoch: 340/500, loss: 2.067489768710362, correct: 50
Epoch: 350/500, loss: 1.9708115800163106, correct: 50
Epoch: 360/500, loss: 1.8337500644454638, correct: 50
Epoch: 370/500, loss: 1.7329315626881154, correct: 50
Epoch: 380/500, loss: 1.6462349035941402, correct: 50
Epoch: 390/500, loss: 1.5599787173299098, correct: 50
Epoch: 400/500, loss: 1.4835950140453935, correct: 50
Epoch: 410/500, loss: 1.4336133458410587, correct: 50
Epoch: 420/500, loss: 1.3487304075180178, correct: 50
Epoch: 430/500, loss: 1.2879570845942223, correct: 50
Epoch: 440/500, loss: 1.2481858652358608, correct: 50
Epoch: 450/500, loss: 1.179687478097186, correct: 50
Epoch: 460/500, loss: 1.1302376868418054, correct: 50
Epoch: 470/500, loss: 1.0958885631031194, correct: 50
Epoch: 480/500, loss: 1.0413714453464988, correct: 50
Epoch: 490/500, loss: 1.0003502527058425, correct: 50
Epoch: 500/500, loss: 0.9618904536928803, correct: 50
```
</details>

### Dataset: Split
Size of hidden layer: 4

0.5 Learning Rate, 600 Epochs

Time per epoch: 0.125s

50 datapoints


![newplot (23)](https://github.com/user-attachments/assets/89a0bad0-a0af-4a19-b46f-49f2ccc447f0)

![newplot (24)](https://github.com/user-attachments/assets/7e417a5c-1d2b-467c-a002-c5075d1b9894)

<details>
<summary>Training Logs Dropdown</summary>

```bash
Epoch: 10/600, loss: 30.98395041572186, correct: 34
Epoch: 20/600, loss: 30.682559095801302, correct: 34
Epoch: 30/600, loss: 30.362241861021843, correct: 34
Epoch: 40/600, loss: 29.95818557165801, correct: 34
Epoch: 50/600, loss: 29.578881839569725, correct: 34
Epoch: 60/600, loss: 29.162255498583367, correct: 34
Epoch: 70/600, loss: 28.66978169710666, correct: 34
Epoch: 80/600, loss: 28.04383038270373, correct: 40
Epoch: 90/600, loss: 26.80591341952736, correct: 41
Epoch: 100/600, loss: 24.865920595050774, correct: 42
Epoch: 110/600, loss: 24.2113672025403, correct: 40
Epoch: 120/600, loss: 23.36517999513341, correct: 38
Epoch: 130/600, loss: 21.067608788043316, correct: 40
Epoch: 140/600, loss: 18.624651370203598, correct: 43
Epoch: 150/600, loss: 19.26458055984833, correct: 45
Epoch: 160/600, loss: 17.84771493921382, correct: 48
Epoch: 170/600, loss: 18.186768556972293, correct: 45
Epoch: 180/600, loss: 13.914349756541442, correct: 47
Epoch: 190/600, loss: 12.46255524037615, correct: 48
Epoch: 200/600, loss: 20.953506682455856, correct: 39
Epoch: 210/600, loss: 16.21884760540959, correct: 45
Epoch: 220/600, loss: 31.86855778889801, correct: 37
Epoch: 230/600, loss: 9.553782039371443, correct: 46
Epoch: 240/600, loss: 17.431818416961555, correct: 42
Epoch: 250/600, loss: 9.119511424359985, correct: 46
Epoch: 260/600, loss: 12.89909045973329, correct: 45
Epoch: 270/600, loss: 12.490329471673054, correct: 44
Epoch: 280/600, loss: 12.424819837350427, correct: 44
Epoch: 290/600, loss: 44.60719194817973, correct: 35
Epoch: 300/600, loss: 5.678712662385299, correct: 49
Epoch: 310/600, loss: 5.083153296172843, correct: 49
Epoch: 320/600, loss: 10.35584348554168, correct: 44
Epoch: 330/600, loss: 43.82577963780868, correct: 35
Epoch: 340/600, loss: 4.771707159835895, correct: 49
Epoch: 350/600, loss: 4.437411453828324, correct: 49
Epoch: 360/600, loss: 5.664554129336168, correct: 48
Epoch: 370/600, loss: 18.37651275180585, correct: 42
Epoch: 380/600, loss: 5.433741252996601, correct: 49
Epoch: 390/600, loss: 3.728281297639507, correct: 50
Epoch: 400/600, loss: 3.605603666635759, correct: 50
Epoch: 410/600, loss: 3.5621741240837186, correct: 50
Epoch: 420/600, loss: 3.4889059984106106, correct: 50
Epoch: 430/600, loss: 3.4135242092360008, correct: 50
Epoch: 440/600, loss: 3.3623485870288308, correct: 50
Epoch: 450/600, loss: 3.2948971136219782, correct: 50
Epoch: 460/600, loss: 3.256756693617032, correct: 50
Epoch: 470/600, loss: 3.1691975669724837, correct: 50
Epoch: 480/600, loss: 3.124575968703847, correct: 50
Epoch: 490/600, loss: 3.1190036340951535, correct: 50
Epoch: 500/600, loss: 3.0473773244937226, correct: 50
Epoch: 510/600, loss: 3.0163901879260617, correct: 50
Epoch: 520/600, loss: 2.9991837405900696, correct: 50
Epoch: 530/600, loss: 2.9804478658627755, correct: 50
Epoch: 540/600, loss: 2.929203170509536, correct: 50
Epoch: 550/600, loss: 2.8986703355783874, correct: 50
Epoch: 560/600, loss: 2.861043566958523, correct: 50
Epoch: 570/600, loss: 2.8244341529957078, correct: 50
Epoch: 580/600, loss: 2.802501959300179, correct: 50
Epoch: 590/600, loss: 2.765204749638425, correct: 50
Epoch: 600/600, loss: 2.7553957028726295, correct: 50
```
</details>

### Dataset: Xor
Size of hidden layer: 6

0.5 Learning Rate, 800 Epochs

Time per epoch: 0.214s

50 datapoints

![newplot (25)](https://github.com/user-attachments/assets/ef56b348-1e13-4489-9ade-ecc62071de94)

![newplot (26)](https://github.com/user-attachments/assets/60082a79-2395-4844-91d5-6b561390c072)

<details>
<summary>Training Logs Dropdown </summary>

```bash
Epoch: 0/800, loss: 0, correct: 0
Epoch: 10/800, loss: 30.451204854722093, correct: 34
Epoch: 20/800, loss: 27.894681383822466, correct: 39
Epoch: 30/800, loss: 25.32909519629715, correct: 42
Epoch: 40/800, loss: 22.933402346934166, correct: 42
Epoch: 50/800, loss: 23.269977404531883, correct: 37
Epoch: 60/800, loss: 22.109624593844412, correct: 38
Epoch: 70/800, loss: 20.996160810374395, correct: 39
Epoch: 80/800, loss: 20.811224487482086, correct: 42
Epoch: 90/800, loss: 19.993790235155462, correct: 41
Epoch: 100/800, loss: 17.59418834718399, correct: 44
Epoch: 110/800, loss: 14.775699638339239, correct: 45
Epoch: 120/800, loss: 14.859783629945545, correct: 44
Epoch: 130/800, loss: 12.220742955098627, correct: 46
Epoch: 140/800, loss: 10.307547383994196, correct: 48
Epoch: 150/800, loss: 8.67034609004117, correct: 48
Epoch: 160/800, loss: 35.48746959772833, correct: 32
Epoch: 170/800, loss: 7.664236298105791, correct: 48
Epoch: 180/800, loss: 6.770587585677321, correct: 48
Epoch: 190/800, loss: 6.3943604351027235, correct: 48
Epoch: 200/800, loss: 19.978502598125758, correct: 33
Epoch: 210/800, loss: 6.434817095825503, correct: 48
Epoch: 220/800, loss: 5.799746406973636, correct: 49
Epoch: 230/800, loss: 7.7765489774238405, correct: 48
Epoch: 240/800, loss: 9.30334297343395, correct: 46
Epoch: 250/800, loss: 5.099055421638423, correct: 48
Epoch: 260/800, loss: 4.637198961768112, correct: 49
Epoch: 270/800, loss: 4.2651304651815485, correct: 49
Epoch: 280/800, loss: 4.190520654451945, correct: 49
Epoch: 290/800, loss: 4.102057241899213, correct: 49
Epoch: 300/800, loss: 3.894867372140891, correct: 49
Epoch: 310/800, loss: 3.69021181965146, correct: 49
Epoch: 320/800, loss: 3.509861939956618, correct: 49
Epoch: 330/800, loss: 3.2688266677681828, correct: 49
Epoch: 340/800, loss: 2.848478853244991, correct: 49
Epoch: 350/800, loss: 2.525196478161843, correct: 50
Epoch: 360/800, loss: 2.3984607015417274, correct: 50
Epoch: 370/800, loss: 2.237126671815052, correct: 50
Epoch: 380/800, loss: 2.0671455009377446, correct: 50
Epoch: 390/800, loss: 1.9249925399032417, correct: 50
Epoch: 400/800, loss: 1.7923931131453597, correct: 50
Epoch: 410/800, loss: 1.5686483087045038, correct: 50
Epoch: 420/800, loss: 1.5175667115947145, correct: 50
Epoch: 430/800, loss: 1.4384966186638788, correct: 50
Epoch: 440/800, loss: 1.3588203330791313, correct: 50
Epoch: 450/800, loss: 1.2853748430772358, correct: 50
Epoch: 460/800, loss: 1.218792115143749, correct: 50
Epoch: 470/800, loss: 1.1614368341866688, correct: 50
Epoch: 480/800, loss: 1.1073136352290678, correct: 50
Epoch: 490/800, loss: 1.0513454789682342, correct: 50
Epoch: 500/800, loss: 1.008724901664503, correct: 50
Epoch: 510/800, loss: 0.9691330604600983, correct: 50
Epoch: 520/800, loss: 0.9322437020186944, correct: 50
Epoch: 530/800, loss: 0.8977934082272815, correct: 50
Epoch: 540/800, loss: 0.8655588388908005, correct: 50
Epoch: 550/800, loss: 0.8359242003126098, correct: 50
Epoch: 560/800, loss: 0.8077716284725854, correct: 50
Epoch: 570/800, loss: 0.7816261173472819, correct: 50
Epoch: 580/800, loss: 0.7570018844291109, correct: 50
Epoch: 590/800, loss: 0.7337344778199121, correct: 50
Epoch: 600/800, loss: 0.7117188414495618, correct: 50
Epoch: 610/800, loss: 0.6908567775828999, correct: 50
Epoch: 620/800, loss: 0.6710621235962083, correct: 50
Epoch: 630/800, loss: 0.6522568456379794, correct: 50
Epoch: 640/800, loss: 0.6343688805981923, correct: 50
Epoch: 650/800, loss: 0.6173365169942548, correct: 50
Epoch: 660/800, loss: 0.6011022399857221, correct: 50
Epoch: 670/800, loss: 0.5856123716497634, correct: 50
Epoch: 680/800, loss: 0.5708214703720031, correct: 50
Epoch: 690/800, loss: 0.5566841046004041, correct: 50
Epoch: 700/800, loss: 0.5431573703967887, correct: 50
Epoch: 710/800, loss: 0.5302071401246156, correct: 50
Epoch: 720/800, loss: 0.5177916816271525, correct: 50
Epoch: 730/800, loss: 0.5058867031643026, correct: 50
Epoch: 740/800, loss: 0.49446075771881154, correct: 50
Epoch: 750/800, loss: 0.4834868097852679, correct: 50
Epoch: 760/800, loss: 0.4729413407168428, correct: 50
Epoch: 770/800, loss: 0.4627962435478587, correct: 50
Epoch: 780/800, loss: 0.4530348027599643, correct: 50
Epoch: 790/800, loss: 0.4436978403001298, correct: 50
Epoch: 800/800, loss: 0.4347483198393946, correct: 50

```

</details>
