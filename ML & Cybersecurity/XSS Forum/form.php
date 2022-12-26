
<!DOCTYPE HTML>  
<html>
<head>
<link rel="stylesheet" href="style.css">
</head>
<body>  

<?php
      // define variables and set to empty values
      $nameErr = $emailErr = $typeErr = $websiteErr = "";
      $name = $email = $type = $comment = $website = "";
      
      if ($_SERVER["REQUEST_METHOD"] == "POST") {
        if (empty($_POST["name"])) {
          $nameErr = "Name is required";
        } else {
          $name = $_POST["name"];
          // check if name only contains letters and whitespace
          if (!preg_match("/^[a-zA-Z-' ]*$/",$name)) {
            $nameErr = "Only letters and white space allowed";
          }
        }
        
        if (empty($_POST["email"])) {
          $emailErr = "Email is required";
        } else {
          $email = $_POST["email"];
          // check if e-mail address is well-formed
          if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            $emailErr = "Invalid email format";
          }
        }
                
        if (empty($_POST["comment"])) {
          $comment = "";
        } else {
          $comment = $_POST["comment"];
        }
      
        if (empty($_POST["type"])) {
          $ypeErr = "Type is required";
        } else {
          $type = $_POST["type"];
        }
      }
      
?>

<style>
    * {
      -webkit-box-sizing: border-box;
      box-sizing: border-box;
    }

    body {
      font-family: 'Open Sans', sans-serif;
      line-height: 1.75em;
      font-size: 16px;
      background-color: black;
      color: #aaa;
    }

    .simple-container {
      max-width: 675px;
      margin: 0 auto;
      padding-top: 70px;
      padding-bottom: 20px;
    }

    .simple-print {
      fill: white;
      stroke: white;
    }
    .simple-print svg {
      height: 100%;
    }

    .simple-close {
      color: white;
      border-color: white;
    }

    .simple-ext-info {
      border-top: 1px solid #aaa;
    }

    p {
      font-size: 16px;
    }

    h1 {
      font-size: 30px;
      line-height: 34px;
    }

    h2 {
      font-size: 20px;
      line-height: 25px;
    }

    h3 {
      font-size: 16px;
      line-height: 27px;
      padding-top: 15px;
      padding-bottom: 15px;
      border-bottom: 1px solid #D8D8D8;
      border-top: 1px solid #D8D8D8;
    }

    hr {
      height: 1px;
      background-color: #d8d8d8;
      border: none;
      width: 100%;
      margin: 0px;
    }

    a[href] {
      color: #1e8ad6;
    }

    a[href]:hover {
      color: #3ba0e6;
    }

    img {
      max-width: 100%;
    }

    li {
      line-height: 1.5em;
    }

    aside,
    [class *= "sidebar"],
    [id *= "sidebar"] {
      max-width: 90%;
      margin: 0 auto;
      border: 1px solid lightgrey;
      padding: 5px 15px;
    }

    @media (min-width: 1921px) {
      body {
        font-size: 18px;
      }
    }

    code,
    .code,
    pre {
      background: black;
      color: green;
      padding: 3px;
      }
    .container {
      display: flex;
      justify-content: center;
    }
    .center {
      width: 800px; 
      padding: 10px;
    }
    .error {color: #FF0000;}

</style>
<h1>A Very Bad Forum</h1>
<div class="container">
      <div class="center">
      <p>Return to the <a href="https://github.com/Nkluge-correa/teeny-tiny_castle" title="aires">castle 🏰</a>.</p>
        <h2>What do you think of this Project? Leave a comment:</h2>
          <p><span class="error">* required field</span></p>
          <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]);?>">  
            Name: <input type="text" name="name" value="<?php echo $name;?>">
            <span class="error">* <?php echo $nameErr;?></span>
            <br><br>
            E-mail: <input type="text" name="email" value="<?php echo $email;?>">
            <span class="error">* <?php echo $emailErr;?></span>
            <br><br>
            Comment: <textarea name="comment" rows="5" cols="40"><?php echo $comment;?></textarea>
            <br><br>
            Ocupation:
            <input type="radio" name="type" <?php if (isset($type) && $type=="Academic") echo "checked";?> value="Academic">Academic
            <input type="radio" name="type" <?php if (isset($type) && $type=="Hobbyist") echo "checked";?> value="Hobbyist">Hobbyist
            <input type="radio" name="type" <?php if (isset($type) && $type=="Other") echo "checked";?> value="Other">Other  
            <span class="error">* <?php echo $typeErr;?></span>
            <br><br>
            <input type="submit" name="submit" value="Submit">  
          </form>
    </div>
</div>
<div class="container">
      <div class="center">
        <?php
        echo "<h2>-- Comment Section -- </h2>";
        echo $name;
        echo "<br>";
        echo $email;
        echo "<br>";
        echo $type;
        echo "<br>";
        echo $comment;
        ?>
    </div>
</div>
</body>
</html>