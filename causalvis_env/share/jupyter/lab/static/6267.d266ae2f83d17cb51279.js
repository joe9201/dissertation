"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[6267],{76267:(e,t,r)=>{r.r(t);r.d(t,{yacas:()=>h});function n(e){var t={},r=e.split(" ");for(var n=0;n<r.length;++n)t[r[n]]=true;return t}var a=n("Assert BackQuote D Defun Deriv For ForEach FromFile "+"FromString Function Integrate InverseTaylor Limit "+"LocalSymbols Macro MacroRule MacroRulePattern "+"NIntegrate Rule RulePattern Subst TD TExplicitSum "+"TSum Taylor Taylor1 Taylor2 Taylor3 ToFile "+"ToStdout ToString TraceRule Until While");var i="(?:(?:\\.\\d+|\\d+\\.\\d*|\\d+)(?:[eE][+-]?\\d+)?)";var o="(?:[a-zA-Z\\$'][a-zA-Z0-9\\$']*)";var u=new RegExp(i);var l=new RegExp(o);var c=new RegExp(o+"?_"+o);var s=new RegExp(o+"\\s*\\(");function f(e,t){var r;r=e.next();if(r==='"'){t.tokenize=p;return t.tokenize(e,t)}if(r==="/"){if(e.eat("*")){t.tokenize=k;return t.tokenize(e,t)}if(e.eat("/")){e.skipToEnd();return"comment"}}e.backUp(1);var n=e.match(/^(\w+)\s*\(/,false);if(n!==null&&a.hasOwnProperty(n[1]))t.scopes.push("bodied");var i=m(t);if(i==="bodied"&&r==="[")t.scopes.pop();if(r==="["||r==="{"||r==="(")t.scopes.push(r);i=m(t);if(i==="["&&r==="]"||i==="{"&&r==="}"||i==="("&&r===")")t.scopes.pop();if(r===";"){while(i==="bodied"){t.scopes.pop();i=m(t)}}if(e.match(/\d+ *#/,true,false)){return"qualifier"}if(e.match(u,true,false)){return"number"}if(e.match(c,true,false)){return"variableName.special"}if(e.match(/(?:\[|\]|{|}|\(|\))/,true,false)){return"bracket"}if(e.match(s,true,false)){e.backUp(1);return"variableName.function"}if(e.match(l,true,false)){return"variable"}if(e.match(/(?:\\|\+|\-|\*|\/|,|;|\.|:|@|~|=|>|<|&|\||_|`|'|\^|\?|!|%|#)/,true,false)){return"operator"}return"error"}function p(e,t){var r,n=false,a=false;while((r=e.next())!=null){if(r==='"'&&!a){n=true;break}a=!a&&r==="\\"}if(n&&!a){t.tokenize=f}return"string"}function k(e,t){var r,n;while((n=e.next())!=null){if(r==="*"&&n==="/"){t.tokenize=f;break}r=n}return"comment"}function m(e){var t=null;if(e.scopes.length>0)t=e.scopes[e.scopes.length-1];return t}const h={name:"yacas",startState:function(){return{tokenize:f,scopes:[]}},token:function(e,t){if(e.eatSpace())return null;return t.tokenize(e,t)},indent:function(e,t,r){if(e.tokenize!==f&&e.tokenize!==null)return null;var n=0;if(t==="]"||t==="];"||t==="}"||t==="};"||t===");")n=-1;return(e.scopes.length+n)*r.unit},languageData:{electricInput:/[{}\[\]()\;]/,commentTokens:{line:"//",block:{open:"/*",close:"*/"}}}}}}]);
//# sourceMappingURL=6267.d266ae2f83d17cb51279.js.map?v=d266ae2f83d17cb51279