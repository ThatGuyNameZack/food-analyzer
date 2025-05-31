
    const input = document.querySelector('input[type="file"]');
    const preview = document.getElementById('preview-image');

    input.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            preview.src = URL.createObjectURL(file);
            preview.style.display = 'block';
        }
    });

