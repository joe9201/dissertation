"use strict";(self["webpackChunk_jupyterlab_application_top"]=self["webpackChunk_jupyterlab_application_top"]||[]).push([[8787],{68787:(e,t,r)=>{r.r(t);r.d(t,{vbScript:()=>a,vbScriptASP:()=>i});function n(e){var t="error";function r(e){return new RegExp("^(("+e.join(")|(")+"))\\b","i")}var n=new RegExp("^[\\+\\-\\*/&\\\\\\^<>=]");var a=new RegExp("^((<>)|(<=)|(>=))");var i=new RegExp("^[\\.,]");var o=new RegExp("^[\\(\\)]");var c=new RegExp("^[A-Za-z][_A-Za-z0-9]*");var u=["class","sub","select","while","if","function","property","with","for"];var l=["else","elseif","case"];var s=["next","loop","wend"];var v=r(["and","or","not","xor","is","mod","eqv","imp"]);var b=["dim","redim","then","until","randomize","byval","byref","new","property","exit","in","const","private","public","get","set","let","stop","on error resume next","on error goto 0","option explicit","call","me"];var d=["true","false","nothing","empty","null"];var f=["abs","array","asc","atn","cbool","cbyte","ccur","cdate","cdbl","chr","cint","clng","cos","csng","cstr","date","dateadd","datediff","datepart","dateserial","datevalue","day","escape","eval","execute","exp","filter","formatcurrency","formatdatetime","formatnumber","formatpercent","getlocale","getobject","getref","hex","hour","inputbox","instr","instrrev","int","fix","isarray","isdate","isempty","isnull","isnumeric","isobject","join","lbound","lcase","left","len","loadpicture","log","ltrim","rtrim","trim","maths","mid","minute","month","monthname","msgbox","now","oct","replace","rgb","right","rnd","round","scriptengine","scriptenginebuildversion","scriptenginemajorversion","scriptengineminorversion","second","setlocale","sgn","sin","space","split","sqr","strcomp","string","strreverse","tan","time","timer","timeserial","timevalue","typename","ubound","ucase","unescape","vartype","weekday","weekdayname","year"];var m=["vbBlack","vbRed","vbGreen","vbYellow","vbBlue","vbMagenta","vbCyan","vbWhite","vbBinaryCompare","vbTextCompare","vbSunday","vbMonday","vbTuesday","vbWednesday","vbThursday","vbFriday","vbSaturday","vbUseSystemDayOfWeek","vbFirstJan1","vbFirstFourDays","vbFirstFullWeek","vbGeneralDate","vbLongDate","vbShortDate","vbLongTime","vbShortTime","vbObjectError","vbOKOnly","vbOKCancel","vbAbortRetryIgnore","vbYesNoCancel","vbYesNo","vbRetryCancel","vbCritical","vbQuestion","vbExclamation","vbInformation","vbDefaultButton1","vbDefaultButton2","vbDefaultButton3","vbDefaultButton4","vbApplicationModal","vbSystemModal","vbOK","vbCancel","vbAbort","vbRetry","vbIgnore","vbYes","vbNo","vbCr","VbCrLf","vbFormFeed","vbLf","vbNewLine","vbNullChar","vbNullString","vbTab","vbVerticalTab","vbUseDefault","vbTrue","vbFalse","vbEmpty","vbNull","vbInteger","vbLong","vbSingle","vbDouble","vbCurrency","vbDate","vbString","vbObject","vbError","vbBoolean","vbVariant","vbDataObject","vbDecimal","vbByte","vbArray"];var p=["WScript","err","debug","RegExp"];var h=["description","firstindex","global","helpcontext","helpfile","ignorecase","length","number","pattern","source","value","count"];var y=["clear","execute","raise","replace","test","write","writeline","close","open","state","eof","update","addnew","end","createobject","quit"];var g=["server","response","request","session","application"];var k=["buffer","cachecontrol","charset","contenttype","expires","expiresabsolute","isclientconnected","pics","status","clientcertificate","cookies","form","querystring","servervariables","totalbytes","contents","staticobjects","codepage","lcid","sessionid","timeout","scripttimeout"];var w=["addheader","appendtolog","binarywrite","end","flush","redirect","binaryread","remove","removeall","lock","unlock","abandon","getlasterror","htmlencode","mappath","transfer","urlencode"];var x=y.concat(h);p=p.concat(m);if(e.isASP){p=p.concat(g);x=x.concat(w,k)}var C=r(b);var I=r(d);var L=r(f);var S=r(p);var D=r(x);var E='"';var j=r(u);var O=r(l);var T=r(s);var z=r(["end"]);var R=r(["do"]);var F=r(["on error resume next","exit"]);var A=r(["rem"]);function B(e,t){t.currentIndent++}function N(e,t){t.currentIndent--}function _(e,r){if(e.eatSpace()){return null}var u=e.peek();if(u==="'"){e.skipToEnd();return"comment"}if(e.match(A)){e.skipToEnd();return"comment"}if(e.match(/^((&H)|(&O))?[0-9\.]/i,false)&&!e.match(/^((&H)|(&O))?[0-9\.]+[a-z_]/i,false)){var l=false;if(e.match(/^\d*\.\d+/i)){l=true}else if(e.match(/^\d+\.\d*/)){l=true}else if(e.match(/^\.\d+/)){l=true}if(l){e.eat(/J/i);return"number"}var s=false;if(e.match(/^&H[0-9a-f]+/i)){s=true}else if(e.match(/^&O[0-7]+/i)){s=true}else if(e.match(/^[1-9]\d*F?/)){e.eat(/J/i);s=true}else if(e.match(/^0(?![\dx])/i)){s=true}if(s){e.eat(/L/i);return"number"}}if(e.match(E)){r.tokenize=W(e.current());return r.tokenize(e,r)}if(e.match(a)||e.match(n)||e.match(v)){return"operator"}if(e.match(i)){return null}if(e.match(o)){return"bracket"}if(e.match(F)){r.doInCurrentLine=true;return"keyword"}if(e.match(R)){B(e,r);r.doInCurrentLine=true;return"keyword"}if(e.match(j)){if(!r.doInCurrentLine)B(e,r);else r.doInCurrentLine=false;return"keyword"}if(e.match(O)){return"keyword"}if(e.match(z)){N(e,r);N(e,r);return"keyword"}if(e.match(T)){if(!r.doInCurrentLine)N(e,r);else r.doInCurrentLine=false;return"keyword"}if(e.match(C)){return"keyword"}if(e.match(I)){return"atom"}if(e.match(D)){return"variableName.special"}if(e.match(L)){return"builtin"}if(e.match(S)){return"builtin"}if(e.match(c)){return"variable"}e.next();return t}function W(e){var t=e.length==1;var r="string";return function(n,a){while(!n.eol()){n.eatWhile(/[^'"]/);if(n.match(e)){a.tokenize=_;return r}else{n.eat(/['"]/)}}if(t){a.tokenize=_}return r}}function q(e,r){var n=r.tokenize(e,r);var a=e.current();if(a==="."){n=r.tokenize(e,r);a=e.current();if(n&&(n.substr(0,8)==="variable"||n==="builtin"||n==="keyword")){if(n==="builtin"||n==="keyword")n="variable";if(x.indexOf(a.substr(1))>-1)n="keyword";return n}else{return t}}return n}return{name:"vbscript",startState:function(){return{tokenize:_,lastToken:null,currentIndent:0,nextLineIndent:0,doInCurrentLine:false,ignoreKeyword:false}},token:function(e,t){if(e.sol()){t.currentIndent+=t.nextLineIndent;t.nextLineIndent=0;t.doInCurrentLine=0}var r=q(e,t);t.lastToken={style:r,content:e.current()};if(r===null)r=null;return r},indent:function(e,t,r){var n=t.replace(/^\s+|\s+$/g,"");if(n.match(T)||n.match(z)||n.match(O))return r.unit*(e.currentIndent-1);if(e.currentIndent<0)return 0;return e.currentIndent*r.unit}}}const a=n({});const i=n({isASP:true})}}]);
//# sourceMappingURL=8787.94119f9e3846afa58ae0.js.map?v=94119f9e3846afa58ae0