
myVar = setTimeout(showPage, 00);
function showPage() {
  document.getElementById("loader").style.display = "none";
//   document.getElementById("myBody").style.display = "block";
  document.getElementsByTagName("Body").style.display="block"
}

function stateChange() {
  // for(var i=0; i<localityOptions.length; i++){
  //   var localityChild = localityOptions[i];
  //   localityOptions.style.color = "red";
  // }
  const localitySelect=document.getElementById("locality")
  var localityOptions=localitySelect.children
  const state=document.getElementById('county').value
  for (let name of localityOptions) {
    name.style.display="none" // main, and then page
  }
  if (state==1){
    const collection = document.getElementsByClassName("nairobi-option");
    for (let name of collection) {
      name.style.display="block" // main, and then page
    }
    localitySelect.value=collection[0].value
  }
  if (state==2){
    const collection = document.getElementsByClassName("kiambu-option");
    for (let name of collection) {
      name.style.display="block" // main, and then page
    }
    localitySelect.value=collection[0].value
  }
  if (state==3){
    const collection = document.getElementsByClassName("kajiado-option");
    for (let name of collection) {
      name.style.display="block" // main, and then page
    }
    localitySelect.value=collection[0].value
  }
  if (state==4){
    const collection = document.getElementsByClassName("mombasa-option");
    for (let name of collection) {
      name.style.display="block" // main, and then page
    }
    localitySelect.value=collection[0].value
  }
  
}
