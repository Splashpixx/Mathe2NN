![alt text](https://i.ibb.co/SQPgRqt/mathe2NN.png)

## Basics

### Eingabeschicht

Die Eingabeschicht ist der Startpunkt des Informationsflusses in einem künstlichen neuronalen Netz.

Eingangssignale werden von den Neuronen am Anfang dieser Schicht aufgenommen und am Ende gewichtet an die Neuronen der ersten Zwischenschicht weitergegeben. Dabei gibt ein Neuron der Eingabeschicht die jeweilige Information an alle Neuronen der ersten Zwischenschicht weiter.

### Zwischenschichten

Zwischen der Eingabe- und der Ausgabeschicht befindet sich in jedem künstlichen neuronalen Netz mindestens eine Zwischenschicht (auch Aktivitätsschicht oder verborgene Schicht von engl.: hidden layer). Je mehr Zwischenschichten es gibt, desto tiefer ist das neuronale Netz, im englischen spricht man daher auch von Deep Learning.

Theoretisch ist die Anzahl der möglichen verborgenen Schichten in einem künstlichen neuronalen Netzwerk unbegrenzt. In der Praxis bewirkt jede hinzukommende verborgene Schicht jedoch auch einen Anstieg der benötigten Rechenleistung, die für den Betrieb des Netzes notwendig ist.

### Ausgabeschicht

Die Ausgabeschicht liegt hinter den Zwischenschichten und bildet die letzte Schicht in einem künstlichen neuronalen Netzwerk. In der Ausgabeschicht angeordnete Neuronen sind jeweils mit allen Neuronen der letzten Zwischenschicht verbunden. Die Ausgabeschicht stellt den Endpunkt des Informationsflusses in einem künstlichen neuronalen Netz dar und enthält das Ergebnis der Informationsverarbeitung durch das Netzwerk.

### Bias

Die Gewichte beschreiben die Intensität des Informationsflusses entlang einer Verbindung in einem neuronalen Netzwerk. Jedes Neuron vergibt dazu ein Gewicht für die durchfließende Information und gibt diese dann gewichtet und nach der Addition eines Wertes für die neuronen-spezifische Verzerrung (Bias) an die Neuronen der nächsten Schicht weiter. Üblicherweise werden die Gewichte und Verzerrungen zum Beginn des Trainings im Wertebereich zwischen -1 und 1 initialisiert, können jedoch später auch deutlich außerhalb dieses Bereichs liegen. Das Ergebnis der Gewichtung und Verzerrung wird oft durch eine sogenannte Aktivierungsfunktion (z.B: Sigmoid oder tanh) geleitet, bevor es an die Neuronen der nächsten Schicht weitergeleitet wird.

## Arten
![alt text](https://jaai.de/wp-content/uploads/2017/09/neuralnetworks.png)
#### [Mehr dazu in der Wiki](https://github.com/Splashpixx/Mathe2NN/wiki#arten-von-neuronalen-netzen)

## Quellen

- https://jaai.de/kuenstliche-neuronale-netze-aufbau-funktion-291/
- https://www.datasciencelearner.com/category/probability/
- https://www.cs.hs-rm.de/~panitz/prog3WS08/perceptron.pdf
- http://yann.lecun.com/exdb/mnist/

### Videos

- https://www.youtube.com/watch?v=oJNHXPs0XDk
- https://www.youtube.com/user/shiffman


### Frameworks für NN und Ai's
[Tensorflow](https://www.tensorflow.org)  
[PyTorch](https://pytorch.org)  
[Keras](https://keras.io)   
[Accord.NET](http://accord-framework.net)   
