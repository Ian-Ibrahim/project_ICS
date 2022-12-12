
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
    name.style.display="none" 
  }
  if (state==1){
    const collection = document.getElementsByClassName("nairobi-option");
    for (let name of collection) {
      name.style.display="block" 
    }
    localitySelect.value=collection[0].value
  }
  if (state==2){
    const collection = document.getElementsByClassName("kiambu-option");
    for (let name of collection) {
      name.style.display="block"
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
function typeChange() {
  // for(var i=0; i<localityOptions.length; i++){
  //   var localityChild = localityOptions[i];
  //   localityOptions.style.color = "red";
  // }
  const subTypeSelect=document.getElementById("sub_type")
  var subTypeOptions=subTypeSelect.children
  const houseType=document.getElementById('type').value
  for (let name of subTypeOptions) {
    name.style.display="none" 
  }
  if (houseType==1){
    const collection = document.getElementsByClassName("house-option");
    for (let name of collection) {
      name.style.display="block"
    }
    subTypeSelect.value=collection[0].value
  }  
  if (houseType==2){
    const collection = document.getElementsByClassName("apartment-option");
    for (let name of collection) {
      name.style.display="block" 
    }
    subTypeSelect.value=collection[0].value
  }
}
function setDate(){
  document.getElementById("year").value=new Date().getFullYear()
  document.getElementById("month").value= new Date().getMonth()
}
function sharedChange(){
  shared=document.getElementById("shared")
  if(shared.checked){
    document.getElementById("sharedValue").value=1
  }else{
    document.getElementById("sharedValue").value=0
  }
}
function furnishedChange(){
  shared=document.getElementById("furnished")
  if(shared.checked){
    document.getElementById("furnishedValue").value=1
  }else{
    document.getElementById("furnishedValue").value=0
  }
}
function servicedChange(){
  shared=document.getElementById("serviced")
  if(shared.checked){
    document.getElementById("servicedValue").value=1
  }else{
    document.getElementById("servicedValue").value=0
  }
}