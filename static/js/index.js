document
  .querySelectorAll(".publication-hero-image img, .publication-image img")
  .forEach((image) => {
  image.addEventListener("error", () => {
    image.closest(".publication-hero-image, .publication-image").classList.add("is-missing");
  });
});
