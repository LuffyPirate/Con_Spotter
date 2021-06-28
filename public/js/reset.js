(function(){

const resetPasswordFunction = () => {
    const resetemail = document.getElementById('re').value;
    const auth = firebase.auth();
    if(resetemail != ""){
    auth.sendPasswordResetEmail(resetemail)
    .then(() => {
        window.alert('Password Reset Email Sent Successfully!');
        window.location.replace('index.html');
    })
    .catch(error => {
        window.alert(error);
    })}
    else{
      window.alert("Please Enter Your Email Address First!")
    }
}


reset.addEventListener('click', resetPasswordFunction);

}());
