
document.addEventListener("DOMContentLoaded", function () {
    const toggle = document.querySelector('.nav-toggle');
    const nav = document.querySelector('.main-nav');

    if (!toggle || !nav) return;

    toggle.addEventListener('click', function () {
        const open = nav.classList.toggle('is-open');

        toggle.setAttribute('aria-expanded', open);
        toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    });
});
document.addEventListener("DOMContentLoaded", function () {
    const viewLiveBtn = document.getElementById("viewLiveBtn");

    viewLiveBtn.addEventListener("click", function (event) {
        event.preventDefault(); 

        window.location.href = "live.html";
    });
});