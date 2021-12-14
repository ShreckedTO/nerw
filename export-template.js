var {name} = {
    w: {weights},
    b: {bias},
    leakyrelu: function(x) {
        return Math.max(x/100,x);
    },
    sigmoid: function(x) {
        return 1/(1+Math.pow(2.71828182846,-x))
    },
    activate: function(){
        var args = [];
        var a = [];
        if (typeof arguments[0] == "object") {
            args = arguments[0];
        } else {
            args = arguments;
        }
{activate}
    }
};
if (typeof exports != "undefined") {
    exports.activate = {name}.activate;
    exports.w = {name}.w;
    exports.b = {name}.b;
    exports.leakyrelu = {name}.leakyrelu;
    exports.sigmoid = {name}.sigmoid;
}