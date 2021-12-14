
# NERW.JS

Welcome to `nerw.js`, a lightweight neural network library. Nerw handles all the tough work for you, so you can forget all the hard work. Nerw is designed to be lightweight, and simple, and is intended for quickly creating artificial intelligence for simple games. You can check out an [example](https://shrecked.my.to/nerw/example) of nerwjs in action!


# Install

```bash
npm install nerwjs
```
# Usage

It's easy to create a simple network with just a few lines of code!
```js
var nerw = require("nerwjs");

// Create a new network instance
var network = new nerw.Network();

// Add inputs to the network
network.addInput("a");
network.addInput("b");

// Add hidden layers
network.addLayer("hiddenLayer", 3);

// Add an output
network.addOutput("result");

// Initiate the network
network.initiate();
```
This doesn't do much by it's self, so create a group of networks:

```js
var networks = new nerw.NetworkGroup(20);

/*
You can treat a network group just like a normal network,
it automatically applies changes to all of the networks!
*/

networks.addInput("a");
networks.addInput("b");

networks.addLayer("hiddenLayerA", 5);
networks.addLayer("hiddenLayerB", 5);

networks.addOutput("result");

networks.initiate();
```
Now with a few networks running, we can start training it on a simple dataset, in this case, an XOR gate.
```js
// simple xor function
function xor(a,b) {
	return (a+b)%2;
}

// it may take some time to get good results
// don't be discouraged if it takes some time
setInterval(function(){
	// reset the score of each network
	networks.each(function(n,i){
		n.score = 0;
	});
	
	// evaluate each of the networks to find the most accurate ones
	for (var i = 0; i < 10; i++) {
		networks.each(function(n,j){
			// create a random input
			var ins = [Math.round(Math.random()),Math.round(Math.random())];
			// get the network's output
			var output = n.activate(ins)[0];
			// get the expected output
			var expected = xor(ins[0],ins[1])*0.5+0.25;
			// add to the score based on the accuracy
			n.score += Math.abs(output-expected);
		});
	}
	networks.sort(); // sort the networks by score
	networks.generation(); // remove the worst networks, and replace them with better ones
	networks.sort(); // sort the networks again

	// see how much your network has improved!
	var ins = [Math.round(Math.random()),Math.round(Math.random())];
	var expected = xor(ins[0],ins[1]);
	var output = (networks.networks[0].activate(ins)[0]-0.25)*2;
	console.log("Expected: " + expected + "\nOutput: " + output);
}, 10);
```
When I was making this example, I found that having the network find values other than 0 and 1 helped, so I replaced 0 with 0.25 and 1 with 0.75, and change it back to 0 and 1 when displaying it on the console.


# Documentation
Not sure what's going on? Check here!
## Methods
### NetworkGroup
``NetworkGroup.addInput``
Adds an input to each of the networks
``NetworkGroup.addLayer``
Adds a hidden layer to each of the networks
``NetworkGroup.addOutput``
Adds an output to each of the networks
``NetworkGroup.initiate``
Initiates the networks
``NetworkGroup.sort``
Sorts the networks by their score (lowest to highest)
``NetworkGroup.generation``
Removes the worst 50% of the networks and replaces them with new networks (higher score is considered worse)
``NetworkGroup.each``
Executes a callback for each of the networks
```js
networks.each(function(n,i){
	// n represents a network
	// use n.score = x to modify the network's score
	n.score += 1;
	// use n.activate to test the ouput of a network
	console.log(n.activate([23,71]));

	// i represents the index of the array
	// not very useful most of the time
	console.log(i);
});
```
### Network
``Network.addInput``
Adds an input to the network
``Network.addLayer``
Adds a hidden layer to the network
``Network.addOutput``
Adds an output to the network
``Network.initiate``
Initiates the network
``Network.activate``
Returns the output of the network given the arguments
``Network.mutate``
Mutates the network
``Network.cloneNetwork``
Returns a clone of the network

# Distribution
Putting your network to use has never been easier! Just use the ``Network.export`` method, and write it's contents to a file. You're already almost there! Now just add your generated file as a script in your html page or require it in node.js, and you can call it from the ``ExportedNetwork.activate`` method.
```js
var fs = require("fs");
fs.writeFileSync("./my_generated_file.js", Network.export("exportName"));
```
### Node.js
```js
var network = require("./my_generated_file.js");
var testValues = [12,64];
console.log(network.activate(testValues));
```
### HTML Script Tag
```html
<script src="./my_generated_file.js"></script>
<script>
	var testValues = [12,64];
	console.log(exportName.activate(testValues));
</script>
```

---

_Tested on Windows 10 (64 bit) with node v15.14.0 and npm v7.7.6_
