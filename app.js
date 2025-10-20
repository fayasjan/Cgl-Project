document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => document.body.classList.remove("fade"), 100);
});
document.getElementById("get-started").addEventListener("click", function () {
  const audioElement = new Audio("Sounds/moan_not_legin.mp3");
  audioElement.play().catch(e => console.log("Audio play error:", e));

  document.body.classList.add("fade");
  setTimeout(() => {
    document.getElementById("main-content").innerHTML = "";
    document.body.classList.remove("fade");
  }, 700);
});



