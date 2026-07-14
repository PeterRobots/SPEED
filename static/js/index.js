document.querySelectorAll(".figure img").forEach((image) => {
  image.addEventListener("error", () => {
    image.closest(".figure").classList.add("is-missing");
  });
});
