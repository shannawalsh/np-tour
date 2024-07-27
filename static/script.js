// Get all custom checkboxes
let customCheckbox = document.querySelectorAll('.custom-checkbox');
let checkbox = document.querySelectorAll('.visually-hidden');

// Add click event listeners to all custom checkboxes
customCheckbox.forEach((customCheckbox, index) => {
  customCheckbox.addEventListener('click', function () {
    this.classList.toggle('checked');
    checkbox[index].checked = !checkbox[index].checked;
  });
});

// Script for drawer
function showDrawer() {
  document.getElementById('drawer').classList.add('open');
  document.getElementById('overlay').classList.add('show');
}

// Script for account dropdown menu
document.addEventListener('click', function (event) {
  var dropdowns = document.querySelectorAll('.dropdown-content');
  dropdowns.forEach(function (dropdown) {
    if (!dropdown.contains(event.target)) {
      dropdown.style.display = 'none';
    }
  });

  if (event.target.matches('.account')) {
    var dropdown = event.target.nextElementSibling;
    if (dropdown.style.display === 'block') {
      dropdown.style.display = 'none';
    } else {
      dropdown.style.display = 'block';
    }
  }
});