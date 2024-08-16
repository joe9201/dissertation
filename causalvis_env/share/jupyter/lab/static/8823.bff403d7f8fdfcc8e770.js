"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[8823,2990],{48823:(n,t,e)=>{e.d(t,{$G:()=>In,$m:()=>An,BB:()=>Un,Ds:()=>ln,Dw:()=>N,EP:()=>a,FP:()=>Nn,HD:()=>En,He:()=>S,Hq:()=>j,IX:()=>L,J_:()=>jn,Jy:()=>On,Kj:()=>kn,Kn:()=>D,N3:()=>F,Oj:()=>o,QA:()=>X,Rg:()=>Hn,TS:()=>vn,TW:()=>wn,We:()=>fn,XW:()=>yn,Xr:()=>pn,ZE:()=>r,ZU:()=>Tn,Zw:()=>$,_k:()=>s,a9:()=>on,ay:()=>B,bM:()=>p,bV:()=>K,cG:()=>E,dH:()=>C,dI:()=>sn,el:()=>u,fE:()=>v,fj:()=>z,hj:()=>Mn,iL:()=>R,id:()=>h,j2:()=>tn,jj:()=>w,jn:()=>dn,k:()=>m,kI:()=>k,kJ:()=>_,kX:()=>b,kg:()=>O,l$:()=>Q,l7:()=>cn,m8:()=>Sn,mJ:()=>W,mK:()=>G,mS:()=>Z,mf:()=>V,nr:()=>hn,qu:()=>nn,rx:()=>Rn,sw:()=>Jn,t7:()=>_n,u5:()=>mn,uU:()=>M,vU:()=>f,vk:()=>xn,yP:()=>zn,yR:()=>g,yb:()=>y,yl:()=>bn});function r(n,t,e){n.fields=t||[];n.fname=e;return n}function u(n){return n==null?null:n.fname}function o(n){return n==null?null:n.fields}function i(n){return n.length===1?l(n[0]):c(n)}const l=n=>function(t){return t[n]};const c=n=>{const t=n.length;return function(e){for(let r=0;r<t;++r){e=e[n[r]]}return e}};function f(n){throw Error(n)}function s(n){const t=[],e=n.length;let r=null,u=0,o="",i,l,c;n=n+"";function s(){t.push(o+n.substring(i,l));o="";i=l+1}for(i=l=0;l<e;++l){c=n[l];if(c==="\\"){o+=n.substring(i,l);o+=n.substring(++l,++l);i=l}else if(c===r){s();r=null;u=-1}else if(r){continue}else if(i===u&&c==='"'){i=l+1;r=c}else if(i===u&&c==="'"){i=l+1;r=c}else if(c==="."&&!u){if(l>i){s()}else{i=l+1}}else if(c==="["){if(l>i)s();u=i=l+1}else if(c==="]"){if(!u)f("Access path missing open bracket: "+n);if(u>0)s();u=0;i=l+1}}if(u)f("Access path missing closing bracket: "+n);if(r)f("Access path missing closing quote: "+n);if(l>i){l++;s()}return t}function a(n,t,e){const u=s(n);n=u.length===1?u[0]:n;return r((e&&e.get||i)(u),[n],t||n)}const h=a("id");const g=r((n=>n),[],"identity");const p=r((()=>0),[],"zero");const b=r((()=>1),[],"one");const y=r((()=>true),[],"true");const m=r((()=>false),[],"false");function d(n,t,e){const r=[t].concat([].slice.call(e));console[n].apply(console,r)}const j=0;const w=1;const M=2;const k=3;const E=4;function O(n,t){let e=arguments.length>2&&arguments[2]!==undefined?arguments[2]:d;let r=n||j;return{level(n){if(arguments.length){r=+n;return this}else{return r}},error(){if(r>=w)e(t||"error","ERROR",arguments);return this},warn(){if(r>=M)e(t||"warn","WARN",arguments);return this},info(){if(r>=k)e(t||"log","INFO",arguments);return this},debug(){if(r>=E)e(t||"log","DEBUG",arguments);return this}}}var _=Array.isArray;function D(n){return n===Object(n)}const A=n=>n!=="__proto__";function v(){for(var n=arguments.length,t=new Array(n),e=0;e<n;e++){t[e]=arguments[e]}return t.reduce(((n,t)=>{for(const e in t){if(e==="signals"){n.signals=x(n.signals,t.signals)}else{const r=e==="legend"?{layout:1}:e==="style"?true:null;R(n,e,t[e],r)}}return n}),{})}function R(n,t,e,r){if(!A(t))return;let u,o;if(D(e)&&!_(e)){o=D(n[t])?n[t]:n[t]={};for(u in e){if(r&&(r===true||r[u])){R(o,u,e[u])}else if(A(u)){o[u]=e[u]}}}else{n[t]=e}}function x(n,t){if(n==null)return t;const e={},r=[];function u(n){if(!e[n.name]){e[n.name]=1;r.push(n)}}t.forEach(u);n.forEach(u);return r}function z(n){return n[n.length-1]}function S(n){return n==null||n===""?null:+n}const J=n=>t=>n*Math.exp(t);const P=n=>t=>Math.log(n*t);const T=n=>t=>Math.sign(t)*Math.log1p(Math.abs(t/n));const U=n=>t=>Math.sign(t)*Math.expm1(Math.abs(t))*n;const H=n=>t=>t<0?-Math.pow(-t,n):Math.pow(t,n);function I(n,t,e,r){const u=e(n[0]),o=e(z(n)),i=(o-u)*t;return[r(u-i),r(o-i)]}function N(n,t){return I(n,t,S,g)}function W(n,t){var e=Math.sign(n[0]);return I(n,t,P(e),J(e))}function X(n,t,e){return I(n,t,H(e),H(1/e))}function $(n,t,e){return I(n,t,T(e),U(e))}function q(n,t,e,r,u){const o=r(n[0]),i=r(z(n)),l=t!=null?r(t):(o+i)/2;return[u(l+(o-l)*e),u(l+(i-l)*e)]}function B(n,t,e){return q(n,t,e,S,g)}function C(n,t,e){const r=Math.sign(n[0]);return q(n,t,e,P(r),J(r))}function G(n,t,e,r){return q(n,t,e,H(r),H(1/r))}function K(n,t,e,r){return q(n,t,e,T(r),U(r))}function Z(n){return 1+~~(new Date(n).getMonth()/3)}function F(n){return 1+~~(new Date(n).getUTCMonth()/3)}function L(n){return n!=null?_(n)?n:[n]:[]}function Q(n,t,e){let r=n[0],u=n[1],o;if(u<r){o=u;u=r;r=o}o=u-r;return o>=e-t?[t,e]:[r=Math.min(Math.max(r,t),e-o),r+o]}function V(n){return typeof n==="function"}const Y="descending";function nn(n,t,e){e=e||{};t=L(t)||[];const u=[],i=[],l={},c=e.comparator||en;L(n).forEach(((n,r)=>{if(n==null)return;u.push(t[r]===Y?-1:1);i.push(n=V(n)?n:a(n,null,e));(o(n)||[]).forEach((n=>l[n]=1))}));return i.length===0?null:r(c(i,u),Object.keys(l))}const tn=(n,t)=>(n<t||n==null)&&t!=null?-1:(n>t||t==null)&&n!=null?1:(t=t instanceof Date?+t:t,n=n instanceof Date?+n:n)!==n&&t===t?-1:t!==t&&n===n?1:0;const en=(n,t)=>n.length===1?rn(n[0],t[0]):un(n,t,n.length);const rn=(n,t)=>function(e,r){return tn(n(e),n(r))*t};const un=(n,t,e)=>{t.push(0);return function(r,u){let o,i=0,l=-1;while(i===0&&++l<e){o=n[l];i=tn(o(r),o(u))}return i*t[l]}};function on(n){return V(n)?n:()=>n}function ln(n,t){let e;return r=>{if(e)clearTimeout(e);e=setTimeout((()=>(t(r),e=null)),n)}}function cn(n){for(let t,e,r=1,u=arguments.length;r<u;++r){t=arguments[r];for(e in t){n[e]=t[e]}}return n}function fn(n,t){let e=0,r,u,o,i;if(n&&(r=n.length)){if(t==null){for(u=n[e];e<r&&(u==null||u!==u);u=n[++e]);o=i=u;for(;e<r;++e){u=n[e];if(u!=null){if(u<o)o=u;if(u>i)i=u}}}else{for(u=t(n[e]);e<r&&(u==null||u!==u);u=t(n[++e]));o=i=u;for(;e<r;++e){u=t(n[e]);if(u!=null){if(u<o)o=u;if(u>i)i=u}}}}return[o,i]}function sn(n,t){const e=n.length;let r=-1,u,o,i,l,c;if(t==null){while(++r<e){o=n[r];if(o!=null&&o>=o){u=i=o;break}}if(r===e)return[-1,-1];l=c=r;while(++r<e){o=n[r];if(o!=null){if(u>o){u=o;l=r}if(i<o){i=o;c=r}}}}else{while(++r<e){o=t(n[r],r,n);if(o!=null&&o>=o){u=i=o;break}}if(r===e)return[-1,-1];l=c=r;while(++r<e){o=t(n[r],r,n);if(o!=null){if(u>o){u=o;l=r}if(i<o){i=o;c=r}}}}return[l,c]}const an=Object.prototype.hasOwnProperty;function hn(n,t){return an.call(n,t)}const gn={};function pn(n){let t={},e;function r(n){return hn(t,n)&&t[n]!==gn}const u={size:0,empty:0,object:t,has:r,get(n){return r(n)?t[n]:undefined},set(n,e){if(!r(n)){++u.size;if(t[n]===gn)--u.empty}t[n]=e;return this},delete(n){if(r(n)){--u.size;++u.empty;t[n]=gn}return this},clear(){u.size=u.empty=0;u.object=t={}},test(n){if(arguments.length){e=n;return u}else{return e}},clean(){const n={};let r=0;for(const u in t){const o=t[u];if(o!==gn&&(!e||!e(o))){n[u]=o;++r}}u.size=r;u.empty=0;u.object=t=n}};if(n)Object.keys(n).forEach((t=>{u.set(t,n[t])}));return u}function bn(n,t,e,r,u,o){if(!e&&e!==0)return o;const i=+e;let l=n[0],c=z(n),f;if(c<l){f=l;l=c;c=f}f=Math.abs(t-l);const s=Math.abs(c-t);return f<s&&f<=i?r:s<=i?u:o}function yn(n,t,e){const r=n.prototype=Object.create(t.prototype);Object.defineProperty(r,"constructor",{value:n,writable:true,enumerable:true,configurable:true});return cn(r,e)}function mn(n,t,e,r){let u=t[0],o=t[t.length-1],i;if(u>o){i=u;u=o;o=i}e=e===undefined||e;r=r===undefined||r;return(e?u<=n:u<n)&&(r?n<=o:n<o)}function dn(n){return typeof n==="boolean"}function jn(n){return Object.prototype.toString.call(n)==="[object Date]"}function wn(n){return n&&V(n[Symbol.iterator])}function Mn(n){return typeof n==="number"}function kn(n){return Object.prototype.toString.call(n)==="[object RegExp]"}function En(n){return typeof n==="string"}function On(n,t,e){if(n){n=t?L(n).map((n=>n.replace(/\\(.)/g,"$1"))):L(n)}const u=n&&n.length,o=e&&e.get||i,l=n=>o(t?[n]:s(n));let c;if(!u){c=function(){return""}}else if(u===1){const t=l(n[0]);c=function(n){return""+t(n)}}else{const t=n.map(l);c=function(n){let e=""+t[0](n),r=0;while(++r<u)e+="|"+t[r](n);return e}}return r(c,n,"key")}function _n(n,t){const e=n[0],r=z(n),u=+t;return!u?e:u===1?r:e+u*(r-e)}const Dn=1e4;function An(n){n=+n||Dn;let t,e,r;const u=()=>{t={};e={};r=0};const o=(u,o)=>{if(++r>n){e=t;t={};r=1}return t[u]=o};u();return{clear:u,has:n=>hn(t,n)||hn(e,n),get:n=>hn(t,n)?t[n]:hn(e,n)?o(n,e[n]):undefined,set:(n,e)=>hn(t,n)?t[n]=e:o(n,e)}}function vn(n,t,e,r){const u=t.length,o=e.length;if(!o)return t;if(!u)return e;const i=r||new t.constructor(u+o);let l=0,c=0,f=0;for(;l<u&&c<o;++f){i[f]=n(t[l],e[c])>0?e[c++]:t[l++]}for(;l<u;++l,++f){i[f]=t[l]}for(;c<o;++c,++f){i[f]=e[c]}return i}function Rn(n,t){let e="";while(--t>=0)e+=n;return e}function xn(n,t,e,r){const u=e||" ",o=n+"",i=t-o.length;return i<=0?o:r==="left"?Rn(u,i)+o:r==="center"?Rn(u,~~(i/2))+o+Rn(u,Math.ceil(i/2)):o+Rn(u,i)}function zn(n){return n&&z(n)-n[0]||0}function Sn(n){return _(n)?"["+n.map(Sn)+"]":D(n)||En(n)?JSON.stringify(n).replace("\u2028","\\u2028").replace("\u2029","\\u2029"):n}function Jn(n){return n==null||n===""?null:!n||n==="false"||n==="0"?false:!!n}const Pn=n=>Mn(n)?n:jn(n)?n:Date.parse(n);function Tn(n,t){t=t||Pn;return n==null||n===""?null:t(n)}function Un(n){return n==null||n===""?null:n+""}function Hn(n){const t={},e=n.length;for(let r=0;r<e;++r)t[n[r]]=true;return t}function In(n,t,e,r){const u=r!=null?r:"…",o=n+"",i=o.length,l=Math.max(0,t-u.length);return i<=t?o:e==="left"?u+o.slice(i-l):e==="center"?o.slice(0,Math.ceil(l/2))+u+o.slice(i-~~(l/2)):o.slice(0,l)+u}function Nn(n,t,e){if(n){if(t){const r=n.length;for(let u=0;u<r;++u){const r=t(n[u]);if(r)e(r,u,n)}}else{n.forEach(e)}}}}}]);
//# sourceMappingURL=8823.bff403d7f8fdfcc8e770.js.map?v=bff403d7f8fdfcc8e770