/**
 * 
 * NERW.js - A lightweight neural network library written in javascript
 * Visit me at https://shrecked.my.to/
 * 
 * @module nerwjs
 * @preserve
 */

if (typeof exports != "undefined") {
    var fs = require("fs");
}

var loadAlerts = false;
if (loadAlerts) console.log("[NERWJS] Loading nerwjs");
var startLoadTime = new Date().getTime();

/**
 * Network node type enum
 * @readonly
 * @enum {Symbol}
 * @preserve
*/
var NodeType = {
    INPUT: Symbol("networkNodeInput"),
    LAYER: Symbol("networkNodeLayer"),
    OUTPUT: Symbol("networkNodeOutput")
};

/**
 * Activation function override enum
 * @readonly
 * @enum {Symbol}
 * @preserve
*/
var ActivationOverride = {
    NONE: Symbol("noneActivationOverride"),
    SIGMOID: Symbol("sigmoidActivationOverride"),
    TANH: Symbol("tanhActivationOverride"),
    RELU: Symbol("ReLUActivationOverride"),
    LEAKYRELU: Symbol("LeakyReLUActivationOverride")
}

/**
 * Sorting algorithm enum
 * @readonly
 * @enum {Symbol}
 * @preserve
*/
var SortingAlgorithmOverride = {
    NONE: Symbol("noneSortingAlgorithmOverride"),
    SELECTION: Symbol("selectionSortingAlgorithmOverride"),
    BUBBLE: Symbol("bubbleSortingAlgorithmOverride")
}

/**
 * Sorting algorithm functions
 * @preserve
 */
var SortingAlgorithm = {
    selectionSort: function(a,b){let c=a.length;for(let d,e=0;e<c;e++){d=e;for(let f=e+1;f<c;f++)a[f][b]<a[d][b]&&(d=f);if(d!=e){let b=a[e];a[e]=a[d],a[d]=b}}return a},
    bubbleSort: function(a,b){let c=a.length;for(let d=0;d<c;d++)for(let d=0;d<c;d++)if(a[d][b]>a[d+1][b]){let b=a[d];a[d]=a[d+1],a[d+1]=b}return a}
};

/**
 * Activation function definitions
 * @preserve
 * */
var ActivationFunction = {
    RELU: function(x) {
        return Math.max(x,0);
    },
    LEAKYRELU: function(x) {
        return Math.max(x,x/100);
    },
    SIGMOID: function(x) {
        return 1/(1+Math.pow(2.71828182846,-x));
    }
}

/**
 * Class representing a single neural network
 * @class
 * @preserve
 */
class Network {
    /**
     * Create a neural network
     * @constructor
     * @preserve
     */
    constructor() {
        this.inputs = [];
        this.layers = [];
        this.outputs = [];
        this.score = 0;
        this.initiated = false;
        this.weightCount = 0;
    }
    
    /**
     * Add an input node
     * @param {string} input - The name of the input node
     * @preserve
     */
    addInput(input) {
        if (this.initiated) return new Error("Network is already initiated");
        if (this.inputs.includes(input)) throw new Error("Network already has input named \"" + input + "\"");
        this.inputs.push(input);
        //console.log("add input " + input);
        this.weightCount++;
    }
    
    /**
     * Add a hidden layer
     * @param {string} layer - The name of the hidden layer
     * @param {number} nc - The ammount of nodes in the hidden layer
     * @preserve
     */
    addLayer(layer,nc) {
        if (this.initiated) return new Error("Network is already initiated");
        if (this.layers.includes(layer)) throw new Error("Network already has layer named \"" + layer + "\"");
        this.layers.push([layer,nc,this.weightCount]);
        this.weightCount = nc;
        //console.log("add layer " + layer + " with " + nc + " nodes");
    }
    
    /**
     * Add an output node
     * @param {string} output - The name of the output node
     * @preserve
     */
    addOutput(output) {
        if (this.initiated) return new Error("Network is already initiated");
        if (this.outputs.includes(output)) throw new Error("Network already has output named \"" + output + "\"");
        this.outputs.push([output,this.weightCount]);
        //console.log("add output " + output);
    }
    
    /**
     * Initiate the neural network - further changes to the network cannot be made
     * @preserve
     */
    initiate() {
        if (this.initiated) return new Error("Network is already initiated");
        for (var i = 0; i < this.layers.length; i++) {
            var createLayer = this.layers[i];
            this.layers[i] = [];
            for (var j = 0; j < createLayer[1]; j++) {
                this.layers[i].push(new NetworkNode(NodeType.LAYER,createLayer[2]));
            }
        }
        for (i = 0; i < this.outputs.length; i++) {
            var createOutput = this.outputs[i];
            this.outputs[i] = new NetworkNode(NodeType.OUTPUT,createOutput[1]);
        }
        this.initiated = true;
        //console.log("initiate");
    }
    
    /**
     * Pass a value into the neural network
     * @param {Object} input - Arguments to pass to the neural network
     * @returns {Object} The output of the network
     * @preserve
     */
    activate() {
        if (arguments.length != this.inputs.length && typeof(arguments[0]) != "object") throw new Error("Bad input count, expected " + this.inputs.length + ", got " + arguments.length);
        if (arguments[0].length != this.inputs.length && typeof(arguments[0]) == "object") throw new Error("Bad input count, expected " + this.inputs.length + ", got " + arguments[0].length);
        var args = [];
        if (typeof(arguments[0]) == "object") args = arguments[0];
        if (typeof(arguments[0]) != "object") args = arguments;
        var layerIn = args;
        var layerOut = [];
        for (var i = 0; i < this.layers.length; i++) {
            for (var j = 0; j < this.layers[i].length; j++) {
                layerOut.push(this.layers[i][j].activate(layerIn));
            }
            layerIn = layerOut;
            layerOut = [];
        }
        for (i = 0; i < this.outputs.length; i++) {
            layerOut.push(this.outputs[i].activate(layerIn));
        }
        return layerOut;
    }
    
    /**
     * Mutate the network
     * @preserve
     */
    mutate() {
        for (var i = 0; i < this.layers.length; i++) {
            for (var j = 0; j < this.layers[i].length; j++) {
                this.layers[i][j].mutate();
            }
        }
        for (i = 0; i < this.outputs.length; i++) {
            this.outputs[i].mutate();
        }
    }
    
    /**
     * Clones the network
     * @returns {Network} Returns the cloned network
     * @preserve
     */
    cloneNetwork() {
        var cloneNetworkObj = new Network();
        cloneNetworkObj.inputs = this.inputs;
        for (var i = 0; i < this.layers.length; i++) {
            cloneNetworkObj.layers[i] = [];
            for (var j = 0; j < this.layers[i].length; j++) {
                cloneNetworkObj.layers[i][j] = this.layers[i][j].cloneNode();
            }
        }
        for (i = 0; i < this.outputs.length; i++) {
            cloneNetworkObj.outputs[i] = this.outputs[i].cloneNode();
        }
        cloneNetworkObj.initiated = this.initiated;
        cloneNetworkObj.score = this.score;
        return cloneNetworkObj;
    }

    /**
     * Generate a javascript string that allows use of the network in other applications
     * @param {string} exportName - Set the name of the exported network
     * @preserve
     */
    export(exportName) {
        if (typeof exports == "undefined") throw new Error("Export function only works in Node.js environment");
        console.warn("The export function is currently experimental");
        var o = fs.readFileSync("./export-template.js").toString();

        var activationExpression = "";
        var weights = [];
        var bias = [];

        // add weights and biases to array from nodes
        for (var i = 0; i < this.layers.length; i++) {
            for (var j = 0; j < this.layers[i].length; j++) {
                for (var k = 0; k < this.layers[i][j].weights.length; k++) {
                    weights.push(this.layers[i][j].weights[k]);
                }
                bias.push(this.layers[i][j].bias);
            }
        }
        for (i = 0; i < this.outputs.length; i++) {
            for (j = 0; j < this.outputs[i].weights.length; j++) {
                weights.push(this.outputs[i].weights[j]);
            }
            bias.push(this.outputs[i].bias);
        }
        // console.log(weights);
        // console.log(bias);

        // TODO: Create function
        // god i dont want to do this its gonna be sooo much work
        // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        var prevLayer = this.inputs;
        var wIndex = 0; // weights index
        var bIndex = 0; // bias index
        var aIndex = 0;
        var aUseIndex = 0;
        var aOffset = 0;

        //create expression in terms of input layer
        // Leaky ReLU
        var activationExpressionAdd = [];
        for (i = 0; i < this.layers[0].length; i++) {
            for (var j = 0; j < this.inputs.length; j++) {
                activationExpressionAdd.push("args[" + j + "]*this.w[" + wIndex + "]");
                wIndex++;
            }
            activationExpressionAdd.push("this.b[" + bIndex + "]");
            bIndex++;
            activationExpression += "\t\ta[" + aIndex + "] = this.leakyrelu(" + activationExpressionAdd.join("+") + ");\n";
            aIndex++;
            activationExpressionAdd = [];
        }
        prevLayer = this.layers[0];

        for (i = 1; i < this.layers.length; i++) {
            // create expression in terms of previous layer
            // Leaky ReLU
            var activationExpressionAdd = [];
            for (j = 0; j < this.layers[i].length; j++) {
                for (var k = 0; k < prevLayer.length; k++) {
                    activationExpressionAdd.push("a[" + (aUseIndex+aOffset) + "]*this.w[" + wIndex + "]");
                    wIndex++;
                    aUseIndex++;
                }
                activationExpressionAdd.push("this.b[" + bIndex + "]");
                bIndex++;
                activationExpression += "\t\ta[" + aIndex + "] = this.leakyrelu(" + activationExpressionAdd.join("+") + ");\n";
                aIndex++;
                activationExpressionAdd = [];
                aUseIndex = 0;
            }
            prevLayer = this.layers[i];
            aOffset += prevLayer.length;
        }
        // create output expression in terms of last hidden layer
        // Sigmoid
        var ret = [];
        var activationExpressionAdd = [];
        var retA = aUseIndex;
        for (i = 0; i < this.outputs.length; i++) {
            for (var j = 0; j < prevLayer.length; j++) {
                activationExpressionAdd.push("a[" + (aUseIndex+aOffset) + "]*this.w[" + wIndex + "]");
                wIndex++;
                aUseIndex++;
            }
            activationExpressionAdd.push("this.b[" + bIndex + "]");
            bIndex++;
            activationExpression += "\t\ta[" + aIndex + "] = this.sigmoid(" + activationExpressionAdd.join("+") + ");\n";
            ret.push("a[" + aIndex + "]");
            aIndex++;
            activationExpressionAdd = [];
            aUseIndex = retA;
        }

        activationExpression += "\t\treturn [" + ret.join(",") + "];";

        o = o.replace(/\{weights\}/, JSON.stringify(weights));
        o = o.replace(/\{bias\}/, JSON.stringify(bias));
        o = o.replace(/\{activate\}/, activationExpression);
        o = o.replace(/\{name\}/g, exportName);
        return o;
    }
}

/**
 * Class representing a network node
 * @class
 * @preserve
 */
class NetworkNode {
    /**
     * Create a network node
     * @param {Symbol} nodeType - The type of the network node
     * @param {number} weightCount - The ammount of weight variables
     * @constructor
     * @preserve
     */
    constructor(nodeType, weightCount) {
        if (!(nodeType == NodeType.INPUT || nodeType == NodeType.LAYER || nodeType == NodeType.OUTPUT)) throw new Error("Unknown node type " + nodeType.toString());
        this.weightCount = weightCount;
        this.type = nodeType;
        this.activationOverride = ActivationOverride.NONE;
        this.weights = [];
        this.bias = Math.random();
        for (var i = 0; i < weightCount; i++) {
            this.weights.push(Math.random());
        }
    }
    
    /**
     * Override the activation function for this node
     * @param {Symbol} activationOverride - The type of activation function to use
     * @preserve
     */
    overrideActivation(activationOverride) {
        for (var i in Object.keys(ActivationOverride)) {
            if (ActivationOverride[Object.keys(ActivationOverride)[i]] == activationOverride) {
                this.activationOverride = activationOverride;
                return;
            }
        }
        throw new Error("Unknown activation override type " + activationOverride.toString());
    }
    
    /**
     * Activate the network node
     * @param {Object} aco - Array of inputs
     * @returns {number} Returns the activated product
     * @preserve
     */
    activate(aco) {
        var ac = 0;
        for (var i in aco) {
            ac += aco[i] * this.weights[i];
        }
        ac += this.bias;
        
        if (this.activationOverride != ActivationOverride.NONE) {
            switch (this.activationOverride) {
                case ActivationOverride.SIGMOID:
                    return ActivationFunction.SIGMOID(ac);
                case ActivationOverride.TANH:
                    return ActivationFunction.TANH(ac);
                case ActivationOverride.RELU:
                    return ActivationFunction.RELU(ac);
                case ActivationOverride.LEAKYRELU:
                    return ActivationFunction.LEAKYRELU(ac);
                default:
                    throw new Error("Unknown activation override type " + this.activationOverride.toString());
            }
        } else {
            switch (this.type) {
                case NodeType.INPUT:
                    return ActivationFunction.RELU(ac);
                case NodeType.LAYER:
                    return ActivationFunction.LEAKYRELU(ac);
                case NodeType.OUTPUT:
                    return ActivationFunction.SIGMOID(ac);
                default:
                    throw new Error("Unknown node type " + this.type.toString());
            }
        }
        //return ac;
    }
    
    /**
     * Mutate the network node
     * @preserve
     */
    mutate() {
        for (var i = 0; i < this.weights.length; i++) {
            this.weights[i] += (Math.random() - 0.5) * 0.1;
        }
        this.bias += (Math.random() - 0.5) * 0.1;
    }

    /**
     * 
     * @param {Object} obj 
     * @returns {Object} Returns the cloned object
     * @preserve
     */
    clone(obj) {
        if (null == obj || "object" != typeof obj) return obj;
        var copy = new obj.constructor();
        for (var attr in obj) {
            if (obj.hasOwnProperty(attr)) copy[attr] = obj[attr];
        }
        return copy;
    }

    /**
     * Clone the network node
     * @returns {NetworkNode} Returns the cloned network node
     * @preserve
     */
    cloneNode() {
        var cloneNodeObj = new NetworkNode(this.type, this.weightCount);
        cloneNodeObj.activationOverride = this.activationOverride;
        cloneNodeObj.weights = this.clone(this.weights);
        cloneNodeObj.bias = this.bias;
        return cloneNodeObj;
    }
}

/**
 * Class representing a group of neural networks
 * @class
 * @preserve
 */
class NetworkGroup {
    /**
     * Create a network group
     * @param {number} amnt - The amount of networks to be created
     * @constructor
     * @preserve
     */
    constructor(amnt) {
        this.networks = [];
        for (var i = 0; i < amnt; i++) {
            this.networks.push(new Network());
        }
        this.initiated = false;
    }

    /**
     * Add an input node to all of the networks
     * @param {string} input - The name of the input node
     * @preserve
     */
    addInput(input) {
        if (this.initiated) throw new Error("Network is already initiated");
        for (var i = 0; i < this.networks.length; i++) {
            this.networks[i].addInput(input);
        }
    }

    /**
     * Add a hidden layer to all of the networks
     * @param {string} layer - The name of the hidden layer
     * @param {number} nc - The ammount of nodes in the hidden layer
     * @preserve
     */
    addLayer(layer,nc) {
        if (this.initiated) throw new Error("Network is already initiated");
        for (var i = 0; i < this.networks.length; i++) {
            this.networks[i].addLayer(layer,nc);
        }
    }

    /**
     * Add an output node to all of the networks
     * @param {string} output - The name of the output node
     * @preserve
     */
    addOutput(output) {
        if (this.initiated) throw new Error("Network is already initiated");
        for (var i = 0; i < this.networks.length; i++) {
            this.networks[i].addOutput(output);
        }
    }

    /**
     * Initiate all of the neural networks - further changes to the networks cannot be made
     * @preserve
     */
    initiate() {
        if (this.initiated) throw new Error("Network is already initiated");
        for (var i = 0; i < this.networks.length; i++) {
            this.networks[i].initiate();
        }
        this.initiated = true;
    }

    /**
     * Execute a function for each of the networks
     * @param {eachCallback} cb - A function to execute for each of the networks
     * @preserve
     */
    each(cb) {
        for (var i = 0; i < this.networks.length; i++) {
            cb(this.networks[i],i);
        }
    }
    /**
     * A callback for each of the networks in a group
     * @callback eachCallback
     * @param {Network} network
     * @param {number} interval
     * @preserve
     */

    /**
     * Sort the networks in the group based on their score
     * @returns {NetworkGroup} The sorted network group
     * @preserve
     */
    sort() {
        this.networks = SortingAlgorithm.selectionSort(this.networks, "score");
        return this.networks;
    }

    /**
     * Replace the worst networks with a mutated version of a better network
     * @preserve
     */
    generation() {
        var offset = Math.round(this.networks.length/2);
        for (var i = 0; i < offset; i++) {
            this.networks[i+offset] = this.networks[i].cloneNetwork();
            this.networks[i+offset].mutate();
        }
    }
}

// If in NodeJS, add to exports
if (typeof exports != "undefined") {
    if (loadAlerts) console.log("[NERWJS] Adding nerwjs to exports");
    exports.NodeType = NodeType;
    exports.ActivationOverride = ActivationOverride;
    exports.SortingAlgorithmOverride = SortingAlgorithmOverride;
    exports.SortingAlgorithm = SortingAlgorithm;
    exports.ActivationFunction = ActivationFunction;
    exports.Network = Network;
    exports.NetworkNode = NetworkNode;
    exports.NetworkGroup = NetworkGroup;
}

if (loadAlerts) console.log("[NERWJS] Loaded nerwjs in " + (new Date().getTime()-startLoadTime) + " ms");