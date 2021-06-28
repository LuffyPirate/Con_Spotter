  const realFileBtn = document.getElementById("real-file");
  const customBtn = document.getElementById("custom-button");
  const customTxt = document.getElementById("custom-text");
  const uploadBtn = document.getElementById("upload");

  function fileValidation(){
  	var fileInput = document.getElementById('real-file');
  	var filePath = fileInput.value;
  	var allowedExtensions = /(\.csv|\.xls)$/i;
      if(!allowedExtensions.exec(filePath)){
          alert('Please upload file having extensions .csv only.');
          fileInput.value = '';
          return false;
  	}else{
          //Image preview
          if (fileInput.files && fileInput.files[0]) {
              var reader = new FileReader();
              reader.onload = function(e) {
                  document.getElementById('real-file').innerHTML = '<img src="'+e.target.result+'"/>';
              };
              reader.readAsDataURL(fileInput.files[0]);
          }
      }

  }

  customBtn.addEventListener("click", function() {
    realFileBtn.click();
  });

  realFileBtn.addEventListener("change", function() {
    if (realFileBtn.value) {
      customTxt.innerHTML = realFileBtn.value.match(
        /[\/\\]([\w\d\s\.\-\(\)]+)$/
      )[1];
    } else {
      customTxt.innerHTML = "No File Chosen, Yet!";
    }
  });

  uploadBtn.addEventListener("click", function(e) {
  		var upfile = document.getElementById('real-file').files[0];
  		var storageRef = firebase.storage().ref("creditcard.csv");
  		var uptask = storageRef.put(upfile);
      var check = firebase.storage.TaskState;
      uptask.on('state_changed',function progress(snapshot){
      let percentage = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
      window.setInterval(function () {},10);
      if (percentage == 100) {
        alert("Upload Completed!");
      }

  });
  });
  function ada(){
    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
      }
    else{
    document.getElementById('loader').style.display = 'flex'
      $.ajax({
        url: "https://conspotterapi.herokuapp.com/ada/",
        type:'GET',
        crossOrigin : true,
        crossDomain : true,

      }).done((response) => {


       document.getElementById('loader').style.display = 'none'
       window.location.assign("ada.html")
     }).fail((err) =>{
        document.getElementById('loader').style.display = 'none'
        alert("Something went wrong!!")
      })
    }

  }
  function rndfrst(){
    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
      }
    else{
      document.getElementById('loader').style.display = 'flex'
      $.ajax({
        url: "https://conspotterapi.herokuapp.com/rforest/",
        type:'GET',
        crossOrigin : true,
        crossDomain : true,

      }).done((response) => {
      document.getElementById('loader').style.display = 'none'

        window.location.assign("rndfrst.html")
      }).fail((err) =>{
      document.getElementById('loader').style.display = 'none'
        alert("Something went wrong!!")
      })
    }

  }
  function knn(){

    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
      }
    else{
      document.getElementById('loader').style.display = 'flex'
      $.ajax({
        url: "https://conspotterapi.herokuapp.com/knn/",
        type:'GET',
        crossOrigin : true,
        crossDomain : true,

      }).done((response) => {

       document.getElementById('loader').style.display = 'none'
        window.location.assign("knn.html")
      }).fail((err) =>{
        document.getElementById('loader').style.display = 'none'
        alert("Something went wrong!!")
      })
    }

  }
  function voting(){

    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
      }
    else{
      document.getElementById('loader').style.display = 'flex'
      $.ajax({
        url: "https://conspotterapi.herokuapp.com/conef/",
        type:'GET',
        crossOrigin : true,
        crossDomain : true,

      }).done((response) => {
       document.getElementById('loader').style.display = 'none'
        window.location.assign("voting.html")
      }).fail((err) =>{
        document.getElementById('loader').style.display = 'none'
        alert("Something went wrong!!")
      })
    }

  }

  /*
  function ada(){
    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
      }
    else{
      window.location.assign("ada.html");
    }
  }

  function rndfrst(){
    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
      }
    else{
      window.location.assign("rndfrst.html");
    }
  }

  function knn(){
    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
    }
    else{
      window.location.assign("knn.html");
    }
  }

  function voting(){
    if(document.getElementById("real-file").value == ''){
        alert('Please upload file ');
      }
    else{
    window.location.assign("voting.html");
    }
  }

  function logout(){
    alert("Do you want to logout?");
    firebase.auth().signOut();
    window.location.replace('index.html');
}
*/
