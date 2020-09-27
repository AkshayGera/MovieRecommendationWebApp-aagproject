   	// For Overlay


   function openNav() {
     document.getElementById("mySidenav").style.width = "220px";
     document.getElementById("main").style.marginLeft = "250px";
     document.body.style.backgroundColor = "rgba(0,0,0,0.4)";
   }

   function closeNav() {
     document.getElementById("mySidenav").style.width = "0";
     document.getElementById("main").style.marginLeft= "0";
     document.body.style.backgroundColor = "white";
   }
     function addGlow1(){
     	document.getElementById("navbox1").classList.add("glow");
     }
     function rmGlow1(){
     	document.getElementById("navbox1").classList.remove("glow");
   }

   function addGlow2(){
     	document.getElementById("navbox2").classList.add("glow");
     }
     function rmGlow2(){
     	document.getElementById("navbox2").classList.remove("glow");
   }
   function addGlow3(){
     	document.getElementById("navbox3").classList.add("glow");
     }
     function rmGlow3(){
     	document.getElementById("navbox3").classList.remove("glow");
   }

   function addGlow4(){
     	document.getElementById("navbox4").classList.add("glow");
     }
     function rmGlow4(){
     	document.getElementById("navbox4").classList.remove("glow");
   }
   function addGlow5(){
     	document.getElementById("navbox5").classList.add("glow");
     }
     function rmGlow5(){
     	document.getElementById("navbox5").classList.remove("glow");
   }

   function ratestar() {
     var a;
     a1 = document.getElementById("indent1");
     a2 = document.getElementById("indent2");
     a3 = document.getElementById("indent3");
     a4 = document.getElementById("indent4");
     a5 = document.getElementById("indent5");
     a1.innerHTML = "&#xf006;";
     a2.innerHTML = "&#xf006;";
     a3.innerHTML = "&#xf006;";
     a4.innerHTML = "&#xf006;";
     a5.innerHTML = "&#xf006;";
     setTimeout(function () {
         a1.innerHTML = "&#xf005;";
       }, 500);
     setTimeout(function () {
         a2.innerHTML = "&#xf005;";
       }, 1000);
     setTimeout(function () {
         a3.innerHTML = "&#xf005;";
       }, 1500);
     setTimeout(function () {
         a4.innerHTML = "&#xf005;";
       }, 2000);
     setTimeout(function () {
         a5.innerHTML = "&#xf005;";
       }, 2500);
   }

   ratestar();
   setInterval(ratestar, 3000);


   function popularity() {
     var a;
     b1 = document.getElementById("pop1");
     b2 = document.getElementById("pop2");
     b3 = document.getElementById("pop3");

     setTimeout(function () {
         b1.innerHTML = "&#xf006;";
       },1000);
     setTimeout(function () {
         b1.innerHTML = "&#xf005;";
       }, 2000);
     setTimeout(function () {
         b2.innerHTML = " + ";
       }, 3000);
     setTimeout(function () {
         b3.innerHTML = "&#xf0E6;";
       }, 4000);
     setTimeout(function () {
         b3.innerHTML = "&#xf086";
       }, 5000);
     setTimeout(function () {
         b1.innerHTML = "";
         b2.innerHTML="";
         b3.innerHTML="";
       }, 6000);
     setTimeout(function () {
         b2.innerHTML = "&#XF251";

       }, 6000);
     setTimeout(function () {
         b2.innerHTML = "&#XF252";

       }, 7000);
     setTimeout(function () {
         b2.innerHTML = "&#XF253";

       }, 8000);
     setTimeout(function () {
         b1.innerHTML = "";
         b2.innerHTML="";
         b3.innerHTML="";
       }, 9000);

   }

   popularity();
   setInterval(popularity, 9000);

