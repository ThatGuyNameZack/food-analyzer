document.addEventListener('DOMContentLoaded', function () {
  const input = document.getElementById('imageInput');
  const preview = document.getElementById('preview');

  input.addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
    } else {
      preview.style.display = 'none';
    }
  });
});
