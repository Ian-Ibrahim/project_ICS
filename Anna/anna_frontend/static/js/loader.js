myVar = setTimeout(showPage, 00);
function showPage() {
  document.getElementById("loader").style.display = "none";
//   document.getElementById("myBody").style.display = "block";
  document.getElementsByTagName("Body").style.display="block"
}