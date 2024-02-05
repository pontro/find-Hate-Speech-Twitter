function toggleMenu() {
    const nav = document.getElementById('mainNav');
    nav.style.display = (nav.style.display === 'block') ? 'none' : 'block';
}
// JavaScript (script.js)

document.addEventListener("DOMContentLoaded", function () {
    const menuButton = document.getElementById('menuButton');
    const mainNav = document.getElementById('mainNav');

    menuButton.addEventListener('click', function () {
        // Toggle the visibility of the menu options
        mainNav.style.display = (mainNav.style.display === 'block') ? 'none' : 'block';
    });
});

console.log('JavaScript is working!');
