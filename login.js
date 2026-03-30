function goToHome(event) {
    event.preventDefault();

    const user = document.querySelector("input[type='text']").value;
    const pass = document.querySelector("input[type='password']").value;

    if (user && pass) {
        window.location.href = "Home.html";
    } else {
        alert("Enter credentials");
    }
}