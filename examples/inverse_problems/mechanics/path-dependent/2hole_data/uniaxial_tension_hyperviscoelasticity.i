begin sierra IsothermalCompression15pcf
  
  title Compression under isothermal conditions
  
  begin feti equation solver feti
  end

  begin gdsw equation solver gdsw
  end
  
  #initial_height = {initial_height = 1} # m
  #initial_area = {initial_area = 1} # m^2

  #load_time = {load_time = 1.0}
  #relax_time = {relax_time = 1.0}
  #end_time = {end_time = load_time + relax_time}

  begin definition for function applied_disp_fun
    type is piecewise linear
    begin values
      0.0          0.0
      {load_time}  1.0
      {load_time + relax_time} 1.0
    end
  end
  
  ######################################################################
  ###                                                                ###
  ### Geometry, Section, and Mesh Definitions                           ###
  ###                                                                ###
  ######################################################################
  
  ### --------------------------------------------------
  ### Directions
  ### --------------------------------------------------
  
  define direction x with vector 1.0 0.0 0.0
  define direction y with vector 0.0 1.0 0.0
  define direction z with vector 0.0 0.0 1.0
  
  ### --------------------------------------------------
  ### Material Model Definitions
  ### --------------------------------------------------
  
  begin property specification for material foam
    density = 240.0

    begin parameters for model modularhyperviscoelasticdamage
      hyper viscoelastic formulation = ISV
      strain energy density          = neohookean
      damage model                   = none
      shift factor model             = none
      neq model                      = quadraticmandel
      #viscous shear rate model       = linear

      bulk modulus                   = 10
      shear modulus                  = 0.855

      num prony terms                = 1
      f2 1                           = 1
      relax time 1                   = 0.25
    end 
  end
  
  ### --------------------------------------------------
  ### Element Block Definitions
  ### --------------------------------------------------

  begin solid section solid_1
    formulation = fully_integrated
  end
  
  begin total lagrange section solid_2
  end

  begin solid section solid_3
  end

  begin finite element model mesh1
    database name = 2holes.g
    Database Type = exodusII
    
    begin parameters for block block_1
      material foam
      model = modularhyperviscoelasticdamage
      section = solid_1
    end
  end
  
  ######################################################################
  ###                                                                ###
  ### ADAGIO Procedure                                               ###
  ###                                                                ###
  ######################################################################
  
  begin adagio procedure adagio_procedure
    
    ### --------------------------------------------------
    ### Time Step Definition
    ### --------------------------------------------------
    
    begin time control
      
      begin time stepping block pload
        start time =  0.0
        begin parameters for adagio region adagio_region
          number of time steps =  11
        end
      end

      begin time stepping block prelax
        start time =  {load_time}
        begin parameters for adagio region adagio_region
          number of time steps =  11
        end
      end
      
      termination time = {end_time}
    end
    
    ######################################################################
    ###                                                                ###
    ### ADAGIO Region                                                  ###
    ###                                                                ###
    ######################################################################
    
    begin adagio region adagio_region
      use finite element model mesh1
      logfile precision = 7
      
      ### --------------------------------------------------
      ### Output Commands
      ### --------------------------------------------------
      
      ######################################################################
      ###                                                                ###
      ### Output Commands                                          ###
      ###                                                                ###
      ######################################################################
      
      
      #User Output
      # The reaction force will be postive in compression for nset_y0
      # Field Output
      begin Results Output output_adagio_region
        Database Name = %B.e
        Database Type = exodusII
        at step 0, increment = 1
        nodal Variables = displacement as displ
        nodal Variables = reaction as reaction_forces
        node variables = force_internal as internal_force
        element Variables = stress as stress
        element variables = shift_factor as shift_factor
        element variables = fv_1 as Fv_1
        element variables = temperature as temperature
        element variables = first_pk_stress
      end
      
      
      # Heartbeat Output
      begin heartbeat ouput DATAHB
        stream name = %B.hb
        append = off
        precision = 10
        legend = on
        format=CSV #default
        labels=off
        timestamp format=""
        at time 0, increment = 1.0E-6
        global time as time
        global timestep as timestep
      end
      
      ### --------------------------------------------------
      ### Boundary Conditions
      ### --------------------------------------------------
      
      begin initial temperature
        block = block_1
        magnitude = 75.0
      end

      # Platen Condition Only During Loading
      begin prescribed displacement
        node set = nodelist_5
        direction = y
        function = applied_disp_fun
        #scale factor = -1.0
      end
      
      # 3 Symmetry Conditions
      begin fixed displacement
        node set = nodelist_3
        components = x y z
      end
      
      begin fixed displacement
        node set = nodelist_5
        components = x z
      end
      
      
      ### --------------------------------------------------
      ### Solver definition
      ### --------------------------------------------------

      begin solver
        begin cg
          reference = belytschko
          target relative residual = 1.0e-6
          target residual = 1.0e-6
          acceptable relative residual = 1.0E-4
          acceptable residual = 1.0E-5
          Maximum Iterations = 5000
          #Line Search secant
          iteration print  = 25
          
          begin full tangent preconditioner
            linear solver = gdsw
            #tangent diagonal scale = 1e-6
            #iteration update = 50
            maximum updates for modelproblem = 5
            small number of iterations = 50
          end
        end
      end 
    end
  end
end
