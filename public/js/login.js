(function(){

login.addEventListener('click',e=>{
const email = document.getElementById('ef').value;
const password = document.getElementById('pf').value;
const auth = firebase.auth();
auth.signInWithEmailAndPassword(email, password).then((result) => {

  window.location.replace('main.html');

})
.catch((error) => {
  var errorCode = error.code;
  var errorMessage = error.message;
  window.alert("Error : " + errorMessage);

});
return true;
})


}());
