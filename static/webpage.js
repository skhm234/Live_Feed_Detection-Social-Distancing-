var btn=document.getElementById("btn");
btn.addEventListener("onClick",function(){
    var image= document.createElement('img');
    image.width="100";
    image.height="100";
    image.src="http://127.0.0.1:5000/result";
    document.body.appendChild(image);
});